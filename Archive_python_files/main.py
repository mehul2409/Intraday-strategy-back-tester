# main.py
# This is the core backtesting engine for your intraday trading system.

import polars as pl
from pathlib import Path
import time
from datetime import time as dt_time
import sys

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
        print("\n" + "="*60)
        print("ERROR: Timezone information not found.")
        print("This program requires timezone data for 'Asia/Kolkata'.")
        print("To fix this, please install the 'tzdata' package by running:")
        print("\n  pip install tzdata\n")
        print("Then, run the script again.")
        print("="*60)
        return False

# --- Configuration ---
# You can adjust these parameters later
STARTING_CAPITAL = 100000.00
BROKERAGE_PERCENT = 0.0003  # Example: 0.03% per trade (adjust to your broker)
SLIPPAGE_PERCENT = 0.0002  # Example: 0.02% slippage on entry/exit

class Backtester:
    """
    A robust, vectorized backtesting engine designed for Indian intraday trading.
    It handles dynamic stops, costs, and specific market timings.
    """

    def __init__(self, starting_capital, brokerage, slippage):
        self.starting_capital = starting_capital
        self.brokerage = brokerage
        self.slippage = slippage
        self.reset_state()

    def reset_state(self):
        """Resets the state of the backtester for a new run."""
        self.cash = self.starting_capital
        self.position = None  # We will only hold one position at a time for simplicity
        self.trade_log = []
        print("-" * 50)
        print(f"Backtester initialized. Starting Capital: ₹{self.starting_capital:,.2f}")
        print("-" * 50)

    def _calculate_atr(self, data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Calculates the Average True Range (ATR)."""
        # Create lagged close
        prev_close = data['close'].shift(1)
        
        # Calculate True Ranges
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - prev_close).abs()
        tr3 = (data['low'] - prev_close).abs()

        # Use pl.max_horizontal to find the row-wise max of the three series
        true_range = pl.max_horizontal(tr1, tr2, tr3)
        
        # Calculate ATR using Exponential Moving Average
        atr = true_range.ewm_mean(span=period, adjust=False)
        
        return data.with_columns(atr.alias(f'atr_{period}'))

    def _apply_slippage(self, price, direction):
        """Applies slippage to a trade price."""
        if direction == 'BUY':
            return price * (1 + self.slippage)
        elif direction == 'SELL':
            return price * (1 - self.slippage)
        return price

    def _log_trade(self, symbol, entry_dt, exit_dt, entry_price, exit_price, qty, pnl, exit_reason):
        """Logs a completed trade."""
        self.trade_log.append({
            "symbol": symbol,
            "entry_datetime": entry_dt,
            "exit_datetime": exit_dt,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": qty,
            "pnl": pnl,
            "exit_reason": exit_reason,
        })
        print(f"{exit_dt} | {exit_reason:<5} | Closed {symbol} at {exit_price:.2f} | PnL: ₹{pnl:,.2f}")


    def run_backtest(self, data: pl.DataFrame, symbol: str, strategy_function):
        """
        The main backtesting loop.

        Args:
            data (pl.DataFrame): A DataFrame with OHLCV data for a single stock.
            symbol (str): The name of the stock being traded.
            strategy_function: A function that takes a data slice and returns 'BUY', 'SELL', or 'HOLD'.
        """
        self.reset_state()
        
        # --- 1. Data Preparation & Indicator Calculation ---
        print(f"Preparing data for {symbol}...")
        
        # Correctly handle datetime conversion and timezone
        date_col_name = 'date' if 'date' in data.columns else 'datetime'
        
        # Check if the column is a string before trying to convert it
        if data[date_col_name].dtype == pl.String:
            data = data.with_columns(
                pl.col(date_col_name).str.to_datetime()
            )
        
        # Ensure timezone is set correctly for all datetime columns
        data = data.with_columns(
            pl.col(date_col_name).dt.replace_time_zone("Asia/Kolkata")
        )

        # Calculate indicators needed for the strategy and exits
        data = self._calculate_atr(data, 14)
        data = data.with_columns([
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(50).alias("sma_50")
        ]).drop_nulls() # Drop initial rows where indicators are null

        # --- 2. The Main Trading Loop ---
        print(f"Starting simulation for {symbol}...")
        
        for row in data.iter_rows(named=True):
            current_dt = row[date_col_name]
            current_price = row['close']
            atr_value = row['atr_14']
            
            # --- Check for exits FIRST ---
            if self.position:
                exit_reason = None
                exit_price = 0
                
                # Rule: Force close at 3:15 PM
                if current_dt.time() >= dt_time(15, 15):
                    exit_reason = "EOD"
                    exit_price = current_price
                # Check for TP hit
                elif current_price >= self.position['tp_price']:
                    exit_reason = "TP"
                    exit_price = self.position['tp_price'] # Assume TP is a limit order
                # Check for Trailing SL hit
                elif current_price <= self.position['trailing_sl']:
                    exit_reason = "TSL"
                    exit_price = self.position['trailing_sl'] # Assume TSL is a stop order
                
                if exit_reason:
                    # Execute the sell
                    exit_price_with_slippage = self._apply_slippage(exit_price, 'SELL')
                    trade_value = exit_price_with_slippage * self.position['quantity']
                    
                    brokerage_cost = trade_value * self.brokerage
                    # In India, STT is on the sell side for intraday
                    stt_cost = trade_value * 0.00025 
                    total_costs = brokerage_cost + self.position['entry_cost'] + stt_cost
                    
                    pnl = (trade_value - self.position['entry_value']) - total_costs
                    self.cash += trade_value - total_costs
                    
                    self._log_trade(symbol, self.position['entry_dt'], current_dt, self.position['entry_price'], exit_price_with_slippage, self.position['quantity'], pnl, exit_reason)
                    self.position = None
                
                # Update Trailing Stop Loss if no exit
                else:
                    self.position['highest_price_since_entry'] = max(self.position['highest_price_since_entry'], row['high'])
                    new_tsl = self.position['highest_price_since_entry'] - (2 * atr_value) # Using 2 * ATR for trailing
                    self.position['trailing_sl'] = max(self.position['trailing_sl'], new_tsl)

            # --- Check for entries SECOND ---
            # Rule: No new trades after 3:00 PM
            if not self.position and current_dt.time() < dt_time(15, 0):
                # We need a slice of data up to the current point for strategy calculation
                # This is inefficient in a loop, but simple to demonstrate.
                # A more optimized version would update indicators incrementally.
                data_slice = data.filter(pl.col(date_col_name) <= current_dt)
                signal = strategy_function(data_slice)

                if signal == 'BUY':
                    # --- Execute Buy ---
                    entry_price_with_slippage = self._apply_slippage(current_price, 'BUY')
                    
                    # Basic position sizing: use 20% of available cash
                    trade_value = self.cash * 0.20
                    quantity = int(trade_value / entry_price_with_slippage)
                    
                    if quantity > 0:
                        entry_cost = (quantity * entry_price_with_slippage) * self.brokerage
                        self.cash -= (quantity * entry_price_with_slippage) + entry_cost

                        # Set dynamic SL and TP using ATR
                        sl_price = current_price - (2 * atr_value)
                        tp_price = current_price + (4 * atr_value) # Maintain 1:2 R:R

                        self.position = {
                            "entry_dt": current_dt,
                            "entry_price": entry_price_with_slippage,
                            "quantity": quantity,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "trailing_sl": sl_price, # Initial TSL is same as SL
                            "highest_price_since_entry": current_price,
                            "entry_value": quantity * entry_price_with_slippage,
                            "entry_cost": entry_cost
                        }
                        print(f"{current_dt} | BUY     | Entry at {entry_price_with_slippage:.2f} for {quantity} shares. SL: {sl_price:.2f}, TP: {tp_price:.2f}")

        print(f"Simulation for {symbol} finished.")
        return self.generate_report()

    def generate_report(self):
        """Generates and prints a performance report from the trade log."""
        if not self.trade_log:
            print("\nNo trades were executed.")
            return None

        log_df = pl.DataFrame(self.trade_log)
        
        # --- Calculate Metrics ---
        total_trades = len(log_df)
        winning_trades = log_df.filter(pl.col('pnl') > 0)
        losing_trades = log_df.filter(pl.col('pnl') <= 0)
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = log_df['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # --- Print Report ---
        print("\n" + "="*50)
        print("          BACKTESTING PERFORMANCE REPORT")
        print("="*50)
        print(f"Final Portfolio Value: ₹{self.cash:,.2f}")
        print(f"Total Net Profit/Loss: ₹{total_pnl:,.2f}")
        print(f"Profit Factor: {abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}" if losing_trades['pnl'].sum() != 0 else "inf")
        print("-" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Winning Trade: ₹{avg_win:,.2f}")
        print(f"Average Losing Trade: ₹{avg_loss:,.2f}")
        print(f"Risk/Reward Ratio (Avg): {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "inf")
        print("="*50)

        return log_df


def moving_average_crossover_strategy(data_slice: pl.DataFrame) -> str:
    """
    A simple pluggable strategy function based on a moving average crossover.
    
    Args:
        data_slice (pl.DataFrame): The historical data up to the current point in time.
    
    Returns:
        str: 'BUY', 'SELL', or 'HOLD'.
    """
    # Get the last two rows to check for a crossover event
    last_two = data_slice.tail(2)
    if len(last_two) < 2:
        return 'HOLD'

    # When getting a single row from a DataFrame, it's still a DataFrame.
    # We need to access the Series, then the value inside it.
    prev_row = last_two[0]
    curr_row = last_two[1]
    
    # FIX: Extract the scalar value from the Series using .item() or [0]
    prev_sma50 = prev_row['sma_50'][0]
    prev_sma20 = prev_row['sma_20'][0]
    curr_sma50 = curr_row['sma_50'][0]
    curr_sma20 = curr_row['sma_20'][0]
    
    # Check for Bullish Crossover (BUY signal)
    # Slow MA was above Fast MA, but is now below.
    if prev_sma50 > prev_sma20 and curr_sma50 < curr_sma20:
        return 'BUY'
        
    # Check for Bearish Crossover (SELL/EXIT signal)
    # Fast MA was above Slow MA, but is now below.
    # Note: Our main engine only acts on 'BUY'. Exits are handled by SL/TP/TSL.
    # This 'SELL' could be used in a shorting strategy.
    if prev_sma50 < prev_sma20 and curr_sma50 > curr_sma20:
        return 'SELL' # For now, our backtester ignores SELL signals for entry.
        
    return 'HOLD'


# --- Main Execution Block ---
if __name__ == "__main__":
    # Add a check for timezone support at the very beginning.
    if not check_timezone_package():
        sys.exit(1) # Exit if the required package is missing

    start_time = time.time()
    
    # Define paths
    parquet_folder = Path("parquet_data")
    log_folder = Path("trade_logs")
    log_folder.mkdir(exist_ok=True)
    
    # Get a list of all stock parquet files
    stock_files = list(parquet_folder.glob("*.parquet"))
    
    if not stock_files:
        print(f"Error: No Parquet files found in '{parquet_folder}'. Please run the conversion script first.")
    else:
        # --- Instantiate the Backtester ---
        my_backtester = Backtester(
            starting_capital=STARTING_CAPITAL,
            brokerage=BROKERAGE_PERCENT,
            slippage=SLIPPAGE_PERCENT
        )

        # --- Loop through each stock file and run the backtest ---
        # For demonstration, let's just run on the first 5 stocks.
        # Remove the '[:5]' to run on all files.
        for stock_file in stock_files[:5]: 
            symbol = stock_file.stem
            print("\n" + "#"*70)
            print(f"      RUNNING BACKTEST FOR: {symbol}")
            print("#"*70)

            # Load the data for the stock
            stock_data = pl.read_parquet(stock_file)
            
            # Run the backtest
            final_log = my_backtester.run_backtest(
                data=stock_data,
                symbol=symbol,
                strategy_function=moving_average_crossover_strategy
            )

            # Save the trade log for this stock to a CSV file
            if final_log is not None and len(final_log) > 0:
                log_file_path = log_folder / f"{symbol}_trade_log.csv"
                final_log.write_csv(log_file_path)
                print(f"Trade log for {symbol} saved to '{log_file_path}'")
            
    end_time = time.time()
    print(f"\nTotal execution time for all backtests: {end_time - start_time:.2f} seconds.")
    print("Backtesting complete! Check the 'trade_logs' folder for results.")
    print("="*70)  