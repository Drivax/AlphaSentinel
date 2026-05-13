"""
AlphaSentinel Utility Module
Helper functions for logging, metrics calculation, and data processing.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

import config

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "AlphaSentinel") -> logging.Logger:
    """
    Initialize logger with both console and file handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (with rotation)
    if config.ENABLE_LOGGING_TO_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=config.LOG_FILE_MAX_BYTES,
            backupCount=config.LOG_FILE_BACKUPS
        )
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Formula: (mean_return - risk_free_rate) / std_dev
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    return float(sharpe) if not np.isnan(sharpe) else 0.0


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (uses only downside volatility).
    
    Formula: (mean_return - risk_free_rate) / downside_std_dev
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    return float(sortino) if not np.isnan(sortino) else 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, str]:
    """
    Calculate maximum drawdown and recovery date.
    
    Args:
        equity_curve: Cumulative portfolio value
        
    Returns:
        Tuple of (max_drawdown as %, recovery_date)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    recovery_date = equity_curve[drawdown == max_dd].index[0].strftime("%Y-%m-%d")
    return float(max_dd), recovery_date


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio = CAGR / Max Drawdown.
    
    Args:
        returns: Daily returns series
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio
    """
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    equity_curve = (1 + returns).cumprod()
    _, _ = calculate_max_drawdown(equity_curve)
    running_max = equity_curve.expanding().max()
    max_dd = ((equity_curve - running_max) / running_max).min()
    
    if max_dd == 0 or max_dd > 0:
        return 0.0
    
    calmar = cagr / abs(max_dd)
    return float(calmar) if not np.isnan(calmar) else 0.0


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """
    Calculate Profit Factor = Sum of Winning Trades / Abs(Sum of Losing Trades).
    
    Args:
        trades: DataFrame with 'pnl' column
        
    Returns:
        Profit factor
    """
    if 'pnl' not in trades.columns or len(trades) == 0:
        return 0.0
    
    winning_sum = trades[trades['pnl'] > 0]['pnl'].sum()
    losing_sum = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    
    if losing_sum == 0:
        return float('inf') if winning_sum > 0 else 0.0
    
    return winning_sum / losing_sum


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate Win Rate = Number of Winning Trades / Total Trades.
    
    Args:
        trades: DataFrame with 'pnl' column
        
    Returns:
        Win rate (0-1)
    """
    if len(trades) == 0:
        return 0.0
    
    winning_trades = len(trades[trades['pnl'] > 0])
    return winning_trades / len(trades)


def calculate_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return (CAGR).
    
    Args:
        returns: Daily returns series
        periods_per_year: Trading periods per year
        
    Returns:
        Annual return as percentage
    """
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    annual = (1 + total_return) ** (1 / years) - 1
    return float(annual)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_technical_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute technical features: returns, volatility, skewness, kurtosis.
    
    Args:
        df: DataFrame with 'Close' column
        lookback: Window for rolling calculations
        
    Returns:
        DataFrame with computed features
    """
    features = df.copy()
    
    # Log returns
    features['log_return'] = np.log(features['Close'] / features['Close'].shift(1))
    
    # Realized volatility (20-day rolling std)
    features['volatility'] = features['log_return'].rolling(lookback).std()
    
    # Skewness (tail risk)
    features['skewness'] = features['log_return'].rolling(lookback).skew()
    
    # Kurtosis (extreme events)
    features['kurtosis'] = features['log_return'].rolling(lookback).kurt()
    
    # Momentum (20-day return)
    features['momentum'] = features['Close'].pct_change(lookback)
    
    # Mean reversion signal (distance from 50-day MA)
    features['ma_50'] = features['Close'].rolling(50).mean()
    features['price_to_ma'] = (features['Close'] - features['ma_50']) / features['ma_50']
    
    # Drop NaN rows
    features = features.dropna()
    
    return features


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Normalize features to [0, 1] range for neural networks.
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to normalize
        
    Returns:
        Tuple of (normalized DataFrame, scaling parameters dict)
    """
    scaling_params = {}
    df_normalized = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            scaling_params[col] = (col_min, col_max)
            
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0.0
    
    return df_normalized, scaling_params


def standardize_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Standardize features to mean=0, std=1.
    
    Args:
        df: DataFrame with features
        feature_cols: List of columns to standardize
        
    Returns:
        Tuple of (standardized DataFrame, scaling parameters dict)
    """
    scaling_params = {}
    df_std = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            scaling_params[col] = (mean, std)
            
            if std != 0:
                df_std[col] = (df[col] - mean) / std
            else:
                df_std[col] = 0.0
    
    return df_std, scaling_params


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_price_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data integrity.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check columns exist
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing required columns. Found: {df.columns.tolist()}")
        return False
    
    # Check no NaN values
    if df[required_cols].isnull().any().any():
        logger.warning("Found NaN values in OHLCV data")
        return False
    
    # Check data consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
    if not (df['High'] >= df['Low']).all():
        logger.warning("High < Low found in data")
        return False
    
    # Check volume > 0
    if not (df['Volume'] > 0).all():
        logger.warning("Zero or negative volume found")
        return False
    
    return True


# ============================================================================
# PERFORMANCE REPORTING
# ============================================================================

def generate_performance_report(
    returns: pd.Series,
    trades: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Generate comprehensive performance metrics report.
    
    Args:
        returns: Daily returns series
        trades: Optional trades DataFrame
        
    Returns:
        Dictionary with all performance metrics
    """
    equity_curve = (1 + returns).cumprod()
    max_dd, _ = calculate_max_drawdown(equity_curve)
    
    report = {
        'annual_return': calculate_annual_return(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'max_drawdown': max_dd,
        'monthly_volatility': returns.resample('M').mean().std() * np.sqrt(252),
        'best_month': returns.resample('M').apply(lambda x: (1 + x).prod() - 1).max(),
        'worst_month': returns.resample('M').apply(lambda x: (1 + x).prod() - 1).min(),
    }
    
    if trades is not None and len(trades) > 0:
        report['profit_factor'] = calculate_profit_factor(trades)
        report['win_rate'] = calculate_win_rate(trades)
    
    return report


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as a readable table."""
    lines = ["\n╔════════════════════════════════════════════════════════╗"]
    lines.append("║            PERFORMANCE METRICS SUMMARY                 ║")
    lines.append("╠════════════════════════════════════════════════════════╣")
    
    for key, value in metrics.items():
        if key in ['annual_return', 'best_month', 'worst_month', 'monthly_volatility', 'max_drawdown']:
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"║ {formatted_key:<40} {value*100:>8.2f}%    ║")
        elif key in ['win_rate']:
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"║ {formatted_key:<40} {value:>8.2%}    ║")
        else:
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"║ {formatted_key:<40} {value:>12.3f}   ║")
    
    lines.append("╚════════════════════════════════════════════════════════╝\n")
    return "\n".join(lines)
