# strategies.py
# This file will contain all of your different trading strategy functions.
# Each function must take a DataFrame and return it with a 'signal' column.

import polars as pl

def moving_average_crossover_strategy(data: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized strategy function: adds indicators and a 'signal' column
    for a moving average crossover.
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
    
    # 4. Calculate ATR needed for exits (the backtester requires this column)
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pl.max_horizontal(tr1, tr2, tr3)
    data = data.with_columns(atr_14=true_range.ewm_mean(span=14, adjust=False))
    
    # 5. Clean up data, ready for the backtester loop
    return data.drop_nulls()


def rsi_oversold_strategy(data: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized RSI strategy: Generates a BUY signal when the RSI crosses
    up from an oversold condition.
    """
    # 1. Calculate RSI
    rsi_period = 14
    oversold_level = 30
    
    delta = data['close'].diff()
    
    # Use clip() with lower_bound and upper_bound arguments
    gain = delta.clip(lower_bound=0)
    loss = delta.clip(upper_bound=0).abs()

    avg_gain = gain.ewm_mean(span=rsi_period, adjust=False)
    avg_loss = loss.ewm_mean(span=rsi_period, adjust=False)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    data = data.with_columns(rsi=rsi)

    # 2. Create signal condition
    # Signal is generated when RSI was below the oversold level on the previous bar
    # and is now above it.
    is_oversold = pl.col("rsi") < oversold_level
    is_recovering = pl.col("rsi") > oversold_level
    
    data = data.with_columns(
        signal=pl.when(is_recovering & is_oversold.shift(1))
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("HOLD"))
    )

    # 3. Calculate ATR needed for exits
    prev_close = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - prev_close).abs()
    tr3 = (data["low"] - prev_close).abs()
    true_range = pl.max_horizontal(tr1, tr2, tr3)
    data = data.with_columns(atr_14=true_range.ewm_mean(span=14, adjust=False))

    # 4. Clean up data
    return data.drop_nulls()

# --- You can add more strategy functions here in the future ---
# def my_new_strategy(data: pl.DataFrame) -> pl.DataFrame:
#     ...
#     return data