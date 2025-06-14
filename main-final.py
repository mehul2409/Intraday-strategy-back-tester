
# main.py
# This is the core backtesting engine for your intraday trading system.
# v4.4: Fixed SyntaxError in HTML report generation.

import multiprocessing
import os
import signal
import sys
import time
from datetime import time as dt_time
from pathlib import Path

import polars as pl

# --- NEW: Import the strategy functions ---
from strategies import moving_average_crossover_strategy, rsi_oversold_strategy


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
    v4.4 generates lightweight HTML summary reports for selectable strategies.
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
            
            # Record equity at each time step
            current_portfolio_value = self.cash + (self.position['quantity'] * current_price if self.position else 0)
            self.equity_curve.append(current_portfolio_value)

        # --- Force-close any open trade at the end ---
        if self.position:
            last_row = prepared_data.tail(1)
            last_price = last_row["close"][0]
            last_dt = last_row["datetime"][0]
            exit_price_slipped = self._apply_slippage(last_price, "SELL")
            trade_value = exit_price_slipped * self.position["quantity"]
            pnl = (trade_value - self.position["entry_value"]) - (
                self.position["entry_cost"] + (trade_value * self.brokerage) + (trade_value * 0.00025)
            )
            self.cash += trade_value
            self._log_trade(symbol, self.position["entry_dt"], last_dt, self.position["entry_price"],
                            exit_price_slipped, self.position["quantity"], pnl, "SIM_END")
            self.position = None

        return pl.DataFrame(self.trade_log) if self.trade_log else None, self.equity_curve

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
    """Generates a lightweight HTML file containing only the performance metrics table."""
    table_rows = ""
    for key, value in metrics.items():
        table_rows += f"<tr><th>{key}</th><td>{value}</td></tr>\n"

    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Backtest Summary: {symbol}</title>
    <style>body{{font-family:sans-serif;margin:40px;background-color:#f4f7f6}} .container{{max-width:600px;margin:auto;background:white;padding:20px;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1)}} table{{width:100%;border-collapse:collapse;margin-top:20px}} th,td{{padding:12px 15px;text-align:left;border-bottom:1px solid #ddd}}</style>
    </head><body><div class="container"><h1>Backtest Summary</h1><h2>Symbol: {symbol}</h2><table>{table_rows}</table></div></body></html>
    """
    # FIX: Corrected syntax error. The 'with open...' part must be outside the f-string.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# --- Worker function for parallel processing 
def run_backtest_for_stock(args):
    """Runs a backtest for a single stock and returns a metrics dictionary."""
    stock_file, strategy_name, strategy_func = args
    symbol = stock_file.stem
    try:
        backtester = Backtester(STARTING_CAPITAL, BROKERAGE_PERCENT, SLIPPAGE_PERCENT)
        
        date_col_name = "date" if "date" in pl.read_parquet_schema(stock_file) else "datetime"
        stock_data = pl.read_parquet(stock_file).rename({date_col_name: "datetime"})
        if stock_data["datetime"].dtype == pl.String:
            stock_data = stock_data.with_columns(pl.col("datetime").str.to_datetime())
        stock_data = stock_data.with_columns(pl.col("datetime").dt.replace_time_zone("Asia/Kolkata"))

        # Explicitly alias the grouping column to avoid errors.
        stock_data_5m = stock_data.group_by_dynamic(
            "datetime",
            every="5m",
            by=pl.col("datetime").dt.date().alias("grouping_date"),
            closed="left",
            label="left",
        ).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum()
        ]).drop("grouping_date") # Drop the temporary column

        trade_log, equity_curve = backtester.run_backtest(stock_data_5m, symbol, strategy_func)
        
        metrics = calculate_performance_metrics(equity_curve, trade_log, STARTING_CAPITAL)
        metrics["symbol"] = symbol
        metrics["strategy"] = strategy_name

        if trade_log is not None and not trade_log.is_empty():
            # Create strategy-specific directories
            log_folder = Path("trade_logs") / strategy_name
            log_folder.mkdir(parents=True, exist_ok=True)
            trade_log.write_csv(log_folder / f"{symbol}_trade_log.csv")
            
            report_folder = Path("html_reports") / strategy_name
            report_folder.mkdir(parents=True, exist_ok=True)
            generate_html_summary(symbol, metrics, report_folder / f"{symbol}_summary.html")

        return metrics
    except Exception as e:
        print(f"!!! Process {os.getpid()} FAILED for {symbol} ({strategy_name}). Error: {e} !!!")
        return {"symbol": symbol, "strategy": strategy_name, "Total Trades": "FAILED"}

def pool_initializer():
    """Ignores Ctrl+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# --- Main Execution Block ---
if __name__ == "__main__":
    if not check_packages():
        sys.exit(1)

    # --- NEW: Strategy Selection ---
    STRATEGIES_TO_RUN = {
        "moving_average_crossover": moving_average_crossover_strategy,
        "rsi_oversold": rsi_oversold_strategy,
    }

    start_time = time.time()
    parquet_folder = Path("parquet_data")
    stock_files = list(parquet_folder.glob("*.parquet"))

    if not stock_files:
        print(f"Error: No Parquet files found in '{parquet_folder}'.")
    else:
        # Loop through each strategy and run the backtest for it
        for strategy_name, strategy_func in STRATEGIES_TO_RUN.items():
            print("\n" + "#" * 70)
            print(f"      RUNNING BACKTESTS FOR STRATEGY: {strategy_name.upper()}")
            print("#" * 70)
            # Limit the number of files for testing purposes
            # Uncomment the next line to limit the number of files for testing
            # files_to_run = stock_files[:12]  # Limit for testing
            # If you want to run all files, comment out the above line and uncomment the next line:
            files_to_run = stock_files
            
            # Create a list of arguments for the worker function
            tasks = [(stock_file, strategy_name, strategy_func) for stock_file in files_to_run]

            num_processes = os.cpu_count()
            print(f"Found {len(files_to_run)} files to test. Using {num_processes} processes.")

            final_results = []
            pool = multiprocessing.Pool(processes=num_processes, initializer=pool_initializer)
            try:
                result_iterator = pool.imap_unordered(run_backtest_for_stock, tasks)
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
                print(f"     SUMMARY FOR STRATEGY: {strategy_name.upper()}")
                print("=" * 50)
                if final_results:
                    summary_df = pl.DataFrame(final_results)
                    print(summary_df)
                    
                    # Save overall summary for this strategy
                    summary_folder = Path("overall_summaries")
                    summary_folder.mkdir(exist_ok=True)
                    summary_df.write_csv(summary_folder / f"{strategy_name}_summary.csv")
                    print(f"\nOverall summary for {strategy_name} saved.")
                else:
                    print("No backtests were completed for this strategy.")
                print("=" * 50)

    end_time = time.time()
    print(f"\nTotal execution time for all strategies: {end_time - start_time:.2f} seconds.")
    print("All backtests completed. Check 'trade_logs' and 'html_reports' directories for results.")
    print("Thank you for using the backtesting engine!")
# End of main.py
# End of main.py