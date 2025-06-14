# main.py
# This is the core backtesting engine for your intraday trading system.
# v3.2: Fixed UnicodeEncodeError by specifying UTF-8 encoding when writing
# HTML reports, ensuring symbols like '₹' are saved correctly.

import multiprocessing
import os
import signal
import sys
import time
from datetime import time as dt_time
from pathlib import Path

import polars as pl


# --- Helper Function to Check for Required Packages ---
def check_packages():
    """Checks for required packages and provides installation instructions."""
    missing_packages = []
    try:
        import zoneinfo
        zoneinfo.ZoneInfo("Asia/Kolkata")
    except (ImportError, zoneinfo.ZoneInfoNotFoundError):
        missing_packages.append("tzdata")

    if missing_packages:
        print("\n" + "=" * 60)
        print("ERROR: Missing required packages.")
        print("To fix this, please install the following package(s):")
        for pkg in missing_packages:
            print(f"\n  pip install {pkg}\n")
        print("Then, run the script again.")
        print("=" * 60)
        return False
    return True


# --- Configuration ---
STARTING_CAPITAL = 100000.00
BROKERAGE_PERCENT = 0.0003
SLIPPAGE_PERCENT = 0.0002


class Backtester:
    """
    A robust, vectorized backtesting engine designed for Indian intraday trading.
    v3.2 generates lightweight HTML summary reports with correct encoding.
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
        self.equity_curve = []

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
        """Main backtesting loop."""
        self.reset_state()
        prepared_data = strategy_function(data)

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
                        + (trade_value * 0.00025)
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
                    new_tsl = self.position["highest_price_since_entry"] - (2 * atr_value)
                    self.position["trailing_sl"] = max(
                        self.position["trailing_sl"], new_tsl
                    )
            
            # --- Handle Entries ---
            if not self.position and row["signal"] == "BUY" and current_dt.time() < dt_time(15, 0):
                entry_price_slipped = self._apply_slippage(current_price, "BUY")
                quantity = int((self.cash * 0.20) / entry_price_slipped)
                if quantity > 0:
                    entry_value = quantity * entry_price_slipped
                    entry_cost = entry_value * self.brokerage
                    self.cash -= entry_value
                    sl_price = current_price - (2 * atr_value)
                    tp_price = current_price + (4 * atr_value)
                    self.position = {
                        "entry_dt": current_dt, "entry_price": entry_price_slipped, "quantity": quantity,
                        "sl_price": sl_price, "tp_price": tp_price, "trailing_sl": sl_price,
                        "highest_price_since_entry": current_price, "entry_value": entry_value, "entry_cost": entry_cost,
                    }
                    print(f"{current_dt} | BUY     | PID: {os.getpid()} | Entry at {entry_price_slipped:.2f} for {quantity} shares")
            
            # Record equity at each time step for drawdown calculation
            current_portfolio_value = self.cash + (self.position['quantity'] * current_price if self.position else 0)
            self.equity_curve.append(current_portfolio_value)

        # --- After loop, check if a position is still open and force-close it ---
        if self.position:
            last_row = prepared_data.tail(1)
            last_price = last_row["close"][0]
            last_dt = last_row["datetime"][0]
            
            print(f"--- Force closing open position for {symbol} at end of simulation ---")
            
            exit_price_slipped = self._apply_slippage(last_price, "SELL")
            trade_value = exit_price_slipped * self.position["quantity"]
            pnl = (trade_value - self.position["entry_value"]) - (
                self.position["entry_cost"]
                + (trade_value * self.brokerage)
                + (trade_value * 0.00025)
            )
            self.cash += trade_value
            self._log_trade(
                symbol,
                self.position["entry_dt"],
                last_dt,
                self.position["entry_price"],
                exit_price_slipped,
                self.position["quantity"],
                pnl,
                "SIM_END",
            )
            self.position = None

        return pl.DataFrame(self.trade_log) if self.trade_log else None, self.equity_curve

def moving_average_crossover_strategy(data: pl.DataFrame) -> pl.DataFrame:
    """Vectorized strategy function: adds indicators and a 'signal' column."""
    data = data.with_columns(
        sma_20=pl.col("close").rolling_mean(20),
        sma_50=pl.col("close").rolling_mean(50),
    )
    crossover = pl.col("sma_20") > pl.col("sma_50")
    crossunder = pl.col("sma_20") < pl.col("sma_50")
    data = data.with_columns(
        signal=pl.when(crossover & crossunder.shift(1)).then(pl.lit("BUY")).otherwise(pl.lit("HOLD"))
    )
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pl.max_horizontal(tr1, tr2, tr3)
    data = data.with_columns(atr_14=true_range.ewm_mean(span=14, adjust=False))
    return data.drop_nulls()

def calculate_performance_metrics(equity_curve, trade_log_df, starting_capital):
    """Calculates key performance metrics from the backtest results."""
    if trade_log_df is None or trade_log_df.is_empty():
        return { "Total Trades": 0 }

    total_trades = len(trade_log_df)
    winning_trades = trade_log_df.filter(pl.col('pnl') > 0)
    losing_trades = trade_log_df.filter(pl.col('pnl') <= 0)
    
    total_pnl = trade_log_df['pnl'].sum()
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    
    # Max Drawdown Calculation
    if not equity_curve:
        max_drawdown = 0
    else:
        equity_series = pl.Series(equity_curve)
        running_max = equity_series.cum_max()
        drawdown = ((equity_series - running_max) / running_max) * 100
        max_drawdown = drawdown.min()
    
    return {
        "Total P&L (₹)": f"{total_pnl:,.2f}",
        "Ending Equity (₹)": f"{equity_curve[-1]:,.2f}" if equity_curve else f"{starting_capital:,.2f}",
        "Total Trades": total_trades,
        "Win Rate (%)": f"{win_rate:.2f}",
        "Max Drawdown (%)": f"{max_drawdown:.2f}",
        "Profit Factor": f"{abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}" if losing_trades['pnl'].sum() != 0 else "inf"
    }

def generate_html_summary(symbol, metrics, output_path):
    """
    Generates a lightweight HTML file containing only the performance metrics table.
    """
    # Create the HTML for the table rows
    table_rows = ""
    for key, value in metrics.items():
        table_rows += f"<tr><th>{key}</th><td>{value}</td></tr>\n"

    # Assemble the full HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtest Summary: {symbol}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
            .container {{ max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a1a1a; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f8f8; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Summary</h1>
            <h2>Symbol: {symbol}</h2>
            <table>
                {table_rows}
            </table>
        </div>
    </body>
    </html>
    """
    
    # FIX: Write the content to the specified file path with UTF-8 encoding
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# --- Worker function for parallel processing ---
def run_backtest_for_stock(stock_file: Path) -> dict:
    """Runs a full backtest and returns a dictionary with performance metrics."""
    symbol = stock_file.stem
    try:
        backtester = Backtester(STARTING_CAPITAL, BROKERAGE_PERCENT, SLIPPAGE_PERCENT)

        date_col_name = "date" if "date" in pl.read_parquet_schema(stock_file) else "datetime"
        stock_data = pl.read_parquet(stock_file).rename({date_col_name: "datetime"})
        if stock_data["datetime"].dtype == pl.String:
            stock_data = stock_data.with_columns(pl.col("datetime").str.to_datetime())
        stock_data = stock_data.with_columns(pl.col("datetime").dt.replace_time_zone("Asia/Kolkata"))

        trade_log, equity_curve = backtester.run_backtest(stock_data, symbol, moving_average_crossover_strategy)
        
        metrics = calculate_performance_metrics(equity_curve, trade_log, STARTING_CAPITAL)
        metrics["symbol"] = symbol

        # --- Generate reports if trades were made ---
        if trade_log is not None and not trade_log.is_empty():
            # Save CSV Log
            log_folder = Path("trade_logs")
            log_folder.mkdir(exist_ok=True)
            trade_log.write_csv(log_folder / f"{symbol}_trade_log.csv")
            
            # Save lightweight HTML Summary
            report_folder = Path("html_reports")
            report_folder.mkdir(exist_ok=True)
            generate_html_summary(symbol, metrics, report_folder / f"{symbol}_summary.html")

        return metrics
    except Exception as e:
        print(f"!!! Process {os.getpid()} FAILED for {symbol}. Error: {e} !!!")
        import traceback
        traceback.print_exc()
        return {"symbol": symbol, "Total Trades": "FAILED"}

def pool_initializer():
    """Ignores Ctrl+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# --- Main Execution Block ---
if __name__ == "__main__":
    if not check_packages():
        sys.exit(1)

    start_time = time.time()
    parquet_folder = Path("parquet_data")
    stock_files = list(parquet_folder.glob("*.parquet"))

    if not stock_files:
        print(f"Error: No Parquet files found in '{parquet_folder}'.")
    else:
        files_to_run = stock_files[:10]
        num_processes = os.cpu_count()
        print(f"Found {len(stock_files)} files. Running on {len(files_to_run)} files using {num_processes} processes.")

        final_results = []
        pool = multiprocessing.Pool(processes=num_processes, initializer=pool_initializer)
        try:
            result_iterator = pool.imap_unordered(run_backtest_for_stock, files_to_run)
            print("\nBacktests running... Press Ctrl+C to interrupt.")
            for result in result_iterator:
                final_results.append(result)
                print(f"Result received: {result}")

        except KeyboardInterrupt:
            print("\n!!! Process interrupted by user. Terminating worker processes... !!!")
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()
        finally:
            print("\n" + "=" * 50)
            print("          OVERALL BACKTEST SUMMARY")
            print("=" * 50)
            if final_results:
                summary_df = pl.DataFrame(final_results)
                print(summary_df)
                
                summary_df.write_csv("overall_summary.csv")
                print("\nOverall summary saved to 'overall_summary.csv'")
            else:
                print("No backtests were completed.")
            print("=" * 50)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print("All processes completed. Exiting main script.")
    sys.exit(0)