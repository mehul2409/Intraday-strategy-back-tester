# main.py
# This is the core backtesting engine for your intraday trading system.
# v2.1: Final consolidated version. Fully vectorized, parallel, and stable.

import multiprocessing
import os
import signal
import sys
import time
from datetime import time as dt_time
from pathlib import Path

import polars as pl


# --- Helper Function to Check for Timezone Support ---
def check_timezone_package():
    """
    Checks if the tzdata package is available, which is required for timezone
    operations on some systems (especially Windows).
    """
    try:
        import zoneinfo

        # Test if the timezone can be found
        zoneinfo.ZoneInfo("Asia/Kolkata")
        return True
    except (ImportError, zoneinfo.ZoneInfoNotFoundError):
        print("\n" + "=" * 60)
        print("ERROR: Timezone information not found.")
        print("This program requires timezone data for 'Asia/Kolkata'.")
        print("To fix this, please install the 'tzdata' package by running:")
        print("\n  pip install tzdata\n")
        print("Then, run the script again.")
        print("=" * 60)
        return False


# --- Configuration ---
STARTING_CAPITAL = 100000.00
BROKERAGE_PERCENT = 0.0003
SLIPPAGE_PERCENT = 0.0002


class Backtester:
    """
    A robust, vectorized backtesting engine designed for Indian intraday trading.
    v2.1 is significantly faster due to vectorized signal generation.
    """

    def __init__(self, starting_capital, brokerage, slippage):
        self.starting_capital = starting_capital
        self.brokerage = brokerage
        self.slippage = slippage
        self.reset_state()

    def reset_state(self):
        """Resets the state for a new run."""
        self.cash = self.starting_capital
        self.position = None
        self.trade_log = []

    def _apply_slippage(self, price, direction):
        """Applies slippage to a trade price."""
        if direction == "BUY":
            return price * (1 + self.slippage)
        elif direction == "SELL":
            return price * (1 - self.slippage)
        return price

    def _log_trade(
        self, symbol, entry_dt, exit_dt, entry_price, exit_price, qty, pnl, exit_reason
    ):
        """Logs a completed trade."""
        self.trade_log.append(
            {
                "symbol": symbol,
                "entry_datetime": entry_dt,
                "exit_datetime": exit_dt,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": qty,
                "pnl": pnl,
                "exit_reason": exit_reason,
            }
        )
        print(
            f"{exit_dt} | {exit_reason:<5} | PID: {os.getpid()} | Closed {symbol} at {exit_price:.2f} | PnL: ₹{pnl:,.2f}"
        )

    def run_backtest(self, data: pl.DataFrame, symbol: str, strategy_function):
        """Main backtesting loop - now much faster."""
        self.reset_state()

        # --- 1. Vectorized Data Preparation ---
        # The strategy function now does all the heavy lifting of data prep
        prepared_data = strategy_function(data)

        # --- 2. The Main Trading Loop ---
        for row in prepared_data.iter_rows(named=True):
            current_dt = row["datetime"]
            current_price = row["close"]
            atr_value = row["atr_14"]

            # --- Handle Exits ---
            if self.position:
                exit_reason, exit_price = None, 0
                if current_dt.time() >= dt_time(15, 15):
                    exit_reason, exit_price = "EOD", current_price
                elif current_price >= self.position["tp_price"]:
                    exit_reason, exit_price = "TP", self.position["tp_price"]
                elif current_price <= self.position["trailing_sl"]:
                    exit_reason, exit_price = "TSL", self.position["trailing_sl"]

                if exit_reason:
                    exit_price_slipped = self._apply_slippage(exit_price, "SELL")
                    trade_value = exit_price_slipped * self.position["quantity"]
                    pnl = (trade_value - self.position["entry_value"]) - (
                        self.position["entry_cost"]
                        + (trade_value * self.brokerage)
                        + (trade_value * 0.00025)  # STT on sell side
                    )
                    self.cash += trade_value
                    self._log_trade(
                        symbol,
                        self.position["entry_dt"],
                        current_dt,
                        self.position["entry_price"],
                        exit_price_slipped,
                        self.position["quantity"],
                        pnl,
                        exit_reason,
                    )
                    self.position = None
                else:
                    self.position["highest_price_since_entry"] = max(
                        self.position["highest_price_since_entry"], row["high"]
                    )
                    new_tsl = self.position["highest_price_since_entry"] - (
                        2 * atr_value
                    )
                    self.position["trailing_sl"] = max(
                        self.position["trailing_sl"], new_tsl
                    )

            # --- Handle Entries ---
            if (
                not self.position
                and row["signal"] == "BUY"
                and current_dt.time() < dt_time(15, 0)
            ):
                entry_price_slipped = self._apply_slippage(current_price, "BUY")
                quantity = int((self.cash * 0.20) / entry_price_slipped)

                if quantity > 0:
                    entry_value = quantity * entry_price_slipped
                    entry_cost = entry_value * self.brokerage
                    self.cash -= entry_value

                    sl_price = current_price - (2 * atr_value)
                    tp_price = current_price + (4 * atr_value)

                    self.position = {
                        "entry_dt": current_dt,
                        "entry_price": entry_price_slipped,
                        "quantity": quantity,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "trailing_sl": sl_price,
                        "highest_price_since_entry": current_price,
                        "entry_value": entry_value,
                        "entry_cost": entry_cost,
                    }
                    print(
                        f"{current_dt} | BUY     | PID: {os.getpid()} | Entry at {entry_price_slipped:.2f} for {quantity} shares"
                    )

        return pl.DataFrame(self.trade_log) if self.trade_log else None


def moving_average_crossover_strategy(data: pl.DataFrame) -> pl.DataFrame:
    """
    A vectorized strategy function. It takes the full DataFrame, adds all
    necessary indicators and a final 'signal' column, and returns it.
    """
    # 1. Calculate indicators
    data = data.with_columns(
        sma_20=pl.col("close").rolling_mean(20),
        sma_50=pl.col("close").rolling_mean(50),
    )

    # 2. Create crossover conditions
    crossover = pl.col("sma_20") > pl.col("sma_50")
    crossunder = pl.col("sma_20") < pl.col("sma_50")

    # 3. Generate signals
    data = data.with_columns(
        signal=pl.when(crossover & crossunder.shift(1))
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("HOLD"))
    )

    # 4. Calculate ATR needed for exits
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pl.max_horizontal(tr1, tr2, tr3)
    data = data.with_columns(atr_14=true_range.ewm_mean(span=14, adjust=False))

    # 5. Clean up data, ready for the backtester loop
    return data.drop_nulls()


# --- Worker function for parallel processing ---
def run_backtest_for_stock(stock_file: Path) -> str:
    """Runs a full backtest for a single stock."""
    symbol = stock_file.stem
    try:
        backtester = Backtester(STARTING_CAPITAL, BROKERAGE_PERCENT, SLIPPAGE_PERCENT)

        # Cleanly handle timezone conversion
        date_col_name = (
            "date" if "date" in pl.read_parquet_schema(stock_file) else "datetime"
        )
        stock_data = pl.read_parquet(stock_file).rename({date_col_name: "datetime"})
        if stock_data["datetime"].dtype == pl.String:
            stock_data = stock_data.with_columns(pl.col("datetime").str.to_datetime())
        stock_data = stock_data.with_columns(
            pl.col("datetime").dt.replace_time_zone("Asia/Kolkata")
        )

        trade_log = backtester.run_backtest(
            stock_data, symbol, moving_average_crossover_strategy
        )

        # Save results
        if trade_log is not None and not trade_log.is_empty():
            log_folder = Path("trade_logs")
            log_folder.mkdir(exist_ok=True)
            trade_log.write_csv(log_folder / f"{symbol}_trade_log.csv")
        return f"{symbol}: Success, {len(trade_log) if trade_log is not None else 0} trades"
    except Exception as e:
        # This will catch errors specific to one stock file and report them
        # without crashing the entire program.
        print(f"!!! Process {os.getpid()} FAILED for {symbol}. Error: {e} !!!")
        return f"{symbol}: FAILED"


def pool_initializer():
    """
    Ignores Ctrl+C in the worker process, letting the main process handle it.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# --- Main Execution Block ---
if __name__ == "__main__":
    if not check_timezone_package():
        sys.exit(1)

    start_time = time.time()
    parquet_folder = Path("parquet_data")
    stock_files = list(parquet_folder.glob("*.parquet"))

    if not stock_files:
        print(f"Error: No Parquet files found in '{parquet_folder}'.")
    else:
        # Set a limit for testing. To run on all files, comment out this line.
        files_to_run = stock_files[:10]

        num_processes = os.cpu_count()
        print(
            f"Found {len(stock_files)} files. Running on {len(files_to_run)} files using {num_processes} processes."
        )

        final_results = []
        # Create a pool of workers with the initializer
        pool = multiprocessing.Pool(processes=num_processes, initializer=pool_initializer)

        try:
            # Use imap_unordered to get results as they are completed
            result_iterator = pool.imap_unordered(run_backtest_for_stock, files_to_run)
            print("\nBacktests running... Press Ctrl+C to interrupt.")

            for result in result_iterator:
                final_results.append(result)
                print(f"Result received: {result}")

        except KeyboardInterrupt:
            print(
                "\n!!! Process interrupted by user. Terminating worker processes... !!!"
            )
            pool.terminate()
            pool.join()

        else:
            # This 'else' block runs only if the try block completes without an exception
            pool.close()
            pool.join()

        finally:
            # This block will always run
            print("\n" + "=" * 50)
            if final_results:
                print(f"PROCESSED {len(final_results)}/{len(files_to_run)} JOBS")
                print("=" * 50)
            else:
                print("No backtests were completed.")
            print("=" * 50)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print("All processes have completed. Exiting main program.")
    sys.exit(0)
# End of main.py
# This code is a complete backtesting engine for intraday trading, designed to be efficient and robust.