"""
AlphaSentinel Data Loader Module
Handles data ingestion from yfinance with caching and preprocessing.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings

import config
from src.utils import logger, validate_price_data

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_price_data(
    symbols: list = None,
    start_date: str = None,
    end_date: str = None,
    use_cache: bool = None
) -> Dict[str, pd.DataFrame]:
    """
    Load historical price data from yfinance with optional caching.
    
    Args:
        symbols: List of ticker symbols (default: config.SYMBOLS)
        start_date: Start date (YYYY-MM-DD format, default: config.START_DATE)
        end_date: End date (YYYY-MM-DD format, default: config.END_DATE)
        use_cache: Use cached data if available (default: config.ENABLE_CACHING)
        
    Returns:
        Dictionary mapping symbol -> OHLCV DataFrame
    """
    # Use defaults
    symbols = symbols or config.SYMBOLS
    start_date = start_date or config.START_DATE
    end_date = end_date or config.END_DATE
    use_cache = use_cache if use_cache is not None else config.ENABLE_CACHING
    
    logger.info(f"Loading price data for {len(symbols)} symbols: {symbols}")
    
    # Try to load from cache
    if use_cache and Path(config.CACHE_FILE).exists():
        try:
            with open(config.CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Validate cache is recent enough
            if 'metadata' in cached_data:
                cache_date = cached_data['metadata'].get('cached_date', '')
                logger.info(f"✓ Loaded from cache (cached date: {cache_date})")
                return cached_data.get('data', {})
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Downloading fresh data...")
    
    # Download from yfinance
    data = {}
    for symbol in symbols:
        try:
            logger.info(f"  Downloading {symbol}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
            
            # Ensure we have a DataFrame, not a Series
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            # Check if data is empty
            if df.empty or len(df) == 0:
                logger.warning(f"  ⚠ No data available for {symbol}")
                continue
            
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # For single symbol downloads, take the first level
                df.columns = df.columns.get_level_values(0)
            
            # Remove rows with zero or negative volume
            if 'Volume' in df.columns:
                df = df[df['Volume'] > 0]
                if len(df) == 0:
                    logger.warning(f"  [SKIP] No rows with positive volume for {symbol}")
                    continue
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Validate data
            if not validate_price_data(df):
                logger.warning(f"  [SKIP] Data validation failed for {symbol}")
                continue
            
            data[symbol] = df
            logger.info(f"[OK] {symbol}: {len(df)} rows")
        
        except Exception as e:
            logger.error(f"  ✗ Failed to download {symbol}: {e}")
            continue
    
    # Cache the data
    if use_cache and data:
        try:
            cache_data = {
                'data': data,
                'metadata': {
                    'cached_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date
                }
            }
            with open(config.CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Data cached to {config.CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    return data


def align_multiasset_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple asset dataframes to common dates.
    
    Args:
        data_dict: Dictionary mapping symbol -> DataFrame
        
    Returns:
        Combined DataFrame with aligned Close prices
    """
    logger.info("Aligning multi-asset data...")
    
    # Extract Close prices and align
    close_prices = {}
    for symbol, df in data_dict.items():
        close_prices[symbol] = df['Close']
    
    combined_df = pd.DataFrame(close_prices)
    combined_df = combined_df.dropna()  # Remove rows with any NaN
    
    logger.info(f"✓ Aligned to {len(combined_df)} common trading days")
    
    return combined_df


def compute_equal_weight_index(close_prices: pd.DataFrame) -> pd.Series:
    """
    Create equal-weight portfolio index.
    
    Args:
        close_prices: DataFrame with Close prices for multiple assets
        
    Returns:
        Equal-weight portfolio returns
    """
    # Normalize prices (1 unit each at start)
    normalized = close_prices / close_prices.iloc[0]
    
    # Equal weight average
    portfolio_value = normalized.mean(axis=1)
    
    return portfolio_value


def compute_single_asset_index(data: pd.DataFrame) -> pd.Series:
    """
    Return price series for single asset.
    
    Args:
        data: DataFrame with 'Close' column
        
    Returns:
        Close price series
    """
    return data['Close']


def split_train_test(
    df: pd.DataFrame,
    train_fraction: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Full dataset
        train_fraction: Fraction for training (0-1)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_fraction)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Data split: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_periods: int,
    test_periods: int,
    step_periods: int = 1
) -> list:
    """
    Create walk-forward analysis splits.
    
    Args:
        df: Full dataset
        train_periods: Number of periods for training
        test_periods: Number of periods for testing
        step_periods: Number of periods to step forward
        
    Returns:
        List of tuples (train_df, test_df)
    """
    splits = []
    idx = 0
    
    while idx + train_periods + test_periods <= len(df):
        train_start = idx
        train_end = idx + train_periods
        test_end = train_end + test_periods
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        
        splits.append((train_df, test_df))
        
        idx += step_periods
    
    logger.info(f"✓ Created {len(splits)} walk-forward splits")
    
    return splits


# ============================================================================
# FEATURE EXTRACTION FOR HMM/LSTM
# ============================================================================

def extract_regime_features(
    price_df: pd.DataFrame,
    lookback: int = config.VOLATILITY_WINDOW
) -> pd.DataFrame:
    """
    Extract features for regime detection (HMM/LSTM input).
    
    Features:
        - Daily log returns
        - 20-day realized volatility
        - 20-day skewness
        - 20-day kurtosis
        - Momentum indicator
        - RSI (14-day Relative Strength Index)
        - MACD histogram (12/26/9 EMA-based)
        - Bollinger Band %B (position within bands)
    
    Args:
        price_df: DataFrame with 'Close' column
        lookback: Window for rolling calculations
        
    Returns:
        DataFrame with features
    """
    features = pd.DataFrame(index=price_df.index)
    close = price_df['Close']

    # Log returns
    features['returns'] = np.log(close / close.shift(1))
    
    # Realized volatility
    features['volatility'] = features['returns'].rolling(lookback).std()
    
    # Skewness (tail risk)
    features['skewness'] = features['returns'].rolling(lookback).skew()
    
    # Excess Kurtosis (extreme events)
    features['kurtosis'] = features['returns'].rolling(lookback).kurt()
    
    # Momentum (20-day return)
    features['momentum'] = close.pct_change(lookback)

    # --- RSI (14-day) ---
    rsi_period = 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    features['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalised to [0, 1]

    # --- MACD histogram (12/26/9) ---
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    # Normalise by price so it is scale-invariant
    features['macd_hist'] = macd_hist / close.replace(0, np.nan)

    # --- Bollinger Band %B (20-day, 2σ) ---
    bb_mid = close.rolling(lookback).mean()
    bb_std = close.rolling(lookback).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    features['bb_pct_b'] = (close - bb_lower) / band_width  # 0 = lower band, 1 = upper band

    # Drop NaN rows
    features = features.dropna()
    
    logger.info(f"✓ Extracted {len(features)} regime feature rows with {len(features.columns)} features")
    
    return features


def create_hmm_sequences(
    features: pd.DataFrame,
    lookback: int = 20
) -> np.ndarray:
    """
    Create sequences for HMM fitting.
    
    Args:
        features: DataFrame with computed features
        lookback: Window size (unused for HMM; for consistency)
        
    Returns:
        Array of shape (n_samples, n_features) for HMM
    """
    feature_cols = ['returns', 'volatility', 'skewness', 'kurtosis', 'momentum',
                    'rsi', 'macd_hist', 'bb_pct_b']
    available_cols = [col for col in feature_cols if col in features.columns]
    
    X = features[available_cols].values
    X = np.nan_to_num(X)  # Replace NaN with 0
    
    logger.info(f"✓ Created HMM sequences: shape {X.shape}")
    
    return X


def create_lstm_sequences(
    features: pd.DataFrame,
    lookback: int = config.LSTM_LOOKBACK
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM.
    
    Args:
        features: DataFrame with features
        lookback: Sequence length
        
    Returns:
        Tuple of (X_sequences, y_targets) for LSTM training
    """
    feature_cols = ['returns', 'volatility', 'skewness', 'kurtosis', 'momentum',
                    'rsi', 'macd_hist', 'bb_pct_b']
    available_cols = [col for col in feature_cols if col in features.columns]
    
    data = features[available_cols].values
    data = np.nan_to_num(data)
    
    X_sequences = []
    y_targets = []
    
    for i in range(len(data) - lookback):
        X_sequences.append(data[i:i + lookback])
        y_targets.append(data[i + lookback, 0])  # Next day return
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    logger.info(f"✓ Created LSTM sequences: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


# ============================================================================
# DATA FORMATTING FOR RL AGENT
# ============================================================================

def format_state_for_rl_agent(
    current_return: float,
    volatility: float,
    skewness: float,
    regime_probs: np.ndarray,
    portfolio_delta: float,
    cash_level: float
) -> np.ndarray:
    """
    Format current market state as RL agent input.
    
    State vector: [return, vol, skew, regime_prob_0, regime_prob_1, regime_prob_2, pos_delta, cash]
    
    Args:
        current_return: Today's return
        volatility: Current volatility
        skewness: Current skewness
        regime_probs: Regime probability vector
        portfolio_delta: Current portfolio delta
        cash_level: Cash as fraction of portfolio
        
    Returns:
        State array of shape (8,) or similar
    """
    state = np.array([
        current_return,
        volatility,
        skewness,
        *regime_probs,
        portfolio_delta,
        cash_level
    ], dtype=np.float32)
    
    return state


# ============================================================================
# DATA UTILITIES
# ============================================================================

def resample_data(df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """
    Resample data to different frequency.
    
    Args:
        df: DataFrame with datetime index
        frequency: Pandas frequency string ('D'=daily, 'W'=weekly, 'M'=monthly)
        
    Returns:
        Resampled DataFrame
    """
    if 'Close' in df.columns:
        return df['Close'].resample(frequency).last().to_frame()
    else:
        return df.resample(frequency).last()


def fill_missing_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing data points.
    
    Args:
        df: DataFrame with potential NaN values
        method: 'ffill' (forward fill), 'bfill' (backward fill)
        
    Returns:
        DataFrame with filled values
    """
    return df.fillna(method=method)


def remove_outliers(df: pd.DataFrame, column: str, std_threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove statistical outliers (beyond N standard deviations).
    
    Args:
        df: DataFrame
        column: Column name to check
        std_threshold: Number of standard deviations
        
    Returns:
        Cleaned DataFrame
    """
    mean = df[column].mean()
    std = df[column].std()
    mask = np.abs(df[column] - mean) <= std_threshold * std
    return df[mask]
