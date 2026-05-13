"""
AlphaSentinel Configuration Module
Centralized parameter management for all trading system components.
"""

import os
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if not exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# European stock market symbols (Yahoo Finance tickers)
SYMBOLS: List[str] = ["^FCHI", "^GDAXI", "^STOXX50E"]  # CAC40, DAX, Euro Stoxx 50
SYMBOL_NAMES = {
    "^FCHI": "CAC40",
    "^GDAXI": "DAX",
    "^STOXX50E": "Euro Stoxx 50"
}

# Historical data range
START_DATE: str = "2015-01-01"
END_DATE: str = "2025-12-31"

# Data cache location
CACHE_FILE: str = str(DATA_DIR / "cached_prices.pkl")

# ============================================================================
# REGIME DETECTION CONFIGURATION
# ============================================================================

# HMM Parameters
HMM_STATES: int = 3  # Number of hidden Markov states (3=calm/volatile/crisis)
HMM_COVARIANCE_TYPE: str = "full"  # Full covariance matrix
HMM_N_ITER: int = 1000  # Max iterations for training
HMM_RANDOM_STATE: int = 42

# LSTM Parameters
LSTM_LOOKBACK: int = 60  # 60 days of historical data
LSTM_UNITS: int = 64  # Units per LSTM layer
LSTM_LAYERS: int = 2  # Number of stacked LSTM layers
LSTM_DROPOUT: float = 0.2  # Dropout rate for regularization
LSTM_EPOCHS: int = 50  # Training epochs
LSTM_BATCH_SIZE: int = 32  # Training batch size
LSTM_VALIDATION_SPLIT: float = 0.2
LSTM_MODEL_PATH: str = str(MODELS_DIR / "lstm_model.h5")

# Isolation Forest (Anomaly Detection)
ISOLATION_FOREST_CONTAMINATION: float = 0.05  # 5% anomaly rate
ISOLATION_FOREST_RANDOM_STATE: int = 42
ANOMALY_ALERT_FLAG: float = 0.3  # Confidence reduction on anomalies

# Feature Engineering
FEATURE_LAGS: List[int] = [1, 5, 10, 20]  # Days for feature lags
VOLATILITY_WINDOW: int = 20  # 20-day rolling volatility
SKEWNESS_WINDOW: int = 20  # Skewness lookback
KURTOSIS_WINDOW: int = 20  # Kurtosis lookback

# ============================================================================
# REINFORCEMENT LEARNING AGENT CONFIGURATION
# ============================================================================

# SAC (Soft Actor-Critic) Parameters
RL_ALGORITHM: str = "SAC"  # Algorithm type
RL_LEARNING_RATE: float = 3e-4  # Actor/Critic learning rate
RL_GAMMA: float = 0.99  # Discount factor (long-term value weight)
RL_TAU: float = 0.005  # Target network update rate
RL_BATCH_SIZE: int = 128  # Training batch size
RL_BUFFER_SIZE: int = 100000  # Replay buffer size
RL_TOTAL_TIMESTEPS: int = 100000  # Total training steps
RL_ENTROPY_COEFF: float = 0.2  # Entropy regularization coefficient
RL_MODEL_PATH: str = str(MODELS_DIR / "rl_agent.pkl")

# Action Space (position sizing)
RL_ACTION_MIN: float = -1.0  # Fully short
RL_ACTION_MAX: float = 1.0  # Fully long
RL_ACTION_THRESHOLD_LONG: float = 0.3  # Threshold for LONG signal
RL_ACTION_THRESHOLD_SHORT: float = -0.3  # Threshold for SHORT signal

# Reward Function Weights
REWARD_ALPHA_SHARPE: float = 1.0  # Sharpe ratio weight
REWARD_ALPHA_MAXDD: float = 2.0  # Max drawdown penalty weight
REWARD_ALPHA_TXNCOST: float = 0.5  # Transaction cost weight
REWARD_ALPHA_TURNOVER: float = 0.1  # Position change penalty
REWARD_SHARPE_LOOKBACK: int = 20  # Days for Sharpe calculation

# ============================================================================
# SIGNAL GENERATION CONFIGURATION
# ============================================================================

# Signal types
SIGNAL_LONG: str = "LONG"
SIGNAL_SHORT: str = "SHORT"
SIGNAL_NEUTRAL: str = "NEUTRAL"

# Confidence thresholds (0-1)
MIN_CONFIDENCE_TO_TRADE: float = 0.55  # Only signal if confidence > 55%

# Regime-dependent position adjustments
REGIME_POSITION_MULTIPLIERS = {
    0: 1.0,   # Calm regime: full sizing
    1: 0.6,   # Volatile regime: 60% sizing
    2: 0.3    # Crisis regime: 30% sizing
}

# ============================================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================================

# Position Sizing
RISK_PER_TRADE: float = 0.02  # Risk 2% per signal
BASELINE_VOLATILITY: float = 0.20  # 20% annualized volatility baseline
VOLATILITY_TARGET: float = 0.10  # Target 10% portfolio volatility

# Hard Stops
INTRADAY_STOP_LOSS: float = -0.015  # -1.5% intraday stop
DAILY_STOP_LOSS: float = -0.02  # -2% daily stop
MAX_DAILY_PORTFOLIO_LOSS: float = 0.03  # Halt new signals if -3% in a day
RECOVERY_THRESHOLD: float = 0.01  # Resume signals at +1% recovery

# Portfolio Limits
MAX_GROSS_EXPOSURE: float = 1.50  # Max 150% total position
MAX_NET_EXPOSURE: float = 1.00  # Max 100% net long
MIN_CORRELATION_THRESHOLD: float = 0.85  # Reduce if > 85% correlated
MAX_SECTOR_CONCENTRATION: float = 0.40  # No sector > 40%

# Transaction Costs
TRANSACTION_COST_BPS: float = 4.0  # 4 basis points per round-trip (0.04%)
SLIPPAGE_BPS: float = 2.0  # Estimated market impact
BORROW_COST_ANNUAL: float = 0.025  # 2.5% annual cost for shorts

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

# Walk-Forward Parameters
WALK_FORWARD_TRAIN_YEARS: int = 3  # 3-year training window
WALK_FORWARD_TEST_MONTHS: int = 6  # 6-month test window
WALK_FORWARD_REBALANCE_FREQ: str = "monthly"  # Retraining frequency
WALK_FORWARD_STEP_MONTHS: int = 1  # 1-month stepping

# Monte Carlo Simulation
MONTE_CARLO_PATHS: int = 1000  # Number of random permutation simulations
MONTE_CARLO_CONFIDENCE_LEVEL: float = 0.95  # Report 5th/95th percentile

# Performance Metrics
RISK_FREE_RATE: float = 0.02  # 2% risk-free rate
MIN_ANNUAL_RETURN: float = 0.08  # Benchmark: passive index return
MIN_SHARPE_RATIO: float = 0.50  # Minimum acceptable Sharpe
MIN_WIN_RATE: float = 0.45  # Minimum acceptable win rate

# Backtesting Results Path
BACKTEST_RESULTS_PATH: str = str(DATA_DIR / "backtest_results.csv")
BACKTEST_TRADES_PATH: str = str(DATA_DIR / "trades_log.csv")

# ============================================================================
# LOGGING & OUTPUT CONFIGURATION
# ============================================================================

# Logging
LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: str = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
LOG_FILE: str = str(LOGS_DIR / "alpha_sentinel.log")
LOG_FILE_MAX_BYTES: int = 10485760  # 10 MB
LOG_FILE_BACKUPS: int = 5  # Keep 5 backup files

# Verbose Mode
VERBOSE: bool = True  # Print progress messages

# ============================================================================
# STREAMLIT DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_THEME: str = "dark"
DASHBOARD_WIDTH: str = "wide"
CHART_HEIGHT: int = 500
CHART_WIDTH: int = 1200

# Update frequency for live dashboard (seconds)
DASHBOARD_REFRESH_INTERVAL: int = 300  # 5 minutes

# ============================================================================
# MODEL TRAINING CONFIGURATION
# ============================================================================

# Retraining schedule
RETRAIN_FREQUENCY_DAYS: int = 63  # Quarterly (63 trading days)
RETRAIN_LOOKBACK_YEARS: int = 5  # Use last 5 years for training

# Train/Test Split
TEST_FRACTION: float = 0.2  # 20% for validation
RANDOM_SEED: int = 42

# ============================================================================
# PAPER TRADING / LIVE SIMULATION
# ============================================================================

# Initial capital
INITIAL_CAPITAL: float = 1_000_000.0  # $1M starting capital

# Trading Parameters
TRADING_HOURS_START: str = "09:00"  # Market open (CET)
TRADING_HOURS_END: str = "17:30"  # Market close (CET)
MAX_HOLDING_PERIOD_DAYS: int = 20  # Max days to hold a position

# ============================================================================
# FEATURE FLAGS
# ============================================================================

USE_PRETRAINED_HMM: bool = False  # Load pre-trained HMM or train fresh
USE_PRETRAINED_LSTM: bool = False  # Load pre-trained LSTM or train fresh
USE_PRETRAINED_RL: bool = False  # Load pre-trained RL agent or train fresh

ENABLE_MONTE_CARLO: bool = True  # Run Monte Carlo simulation
ENABLE_ANOMALY_DETECTION: bool = True  # Use Isolation Forest
ENABLE_LOGGING_TO_FILE: bool = True  # Write logs to file
ENABLE_CACHING: bool = True  # Cache yfinance data

# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def get_config_summary() -> str:
    """Return a formatted summary of key configuration parameters."""
    summary = f"""
    ╔══════════════════════════════════════════════════════╗
    ║         AlphaSentinel Configuration Summary         ║
    ╚══════════════════════════════════════════════════════╝
    
    📊 DATA:
       Symbols: {', '.join([SYMBOL_NAMES.get(s, s) for s in SYMBOLS])}
       Period: {START_DATE} to {END_DATE}
    
    🔍 REGIME DETECTION:
       HMM States: {HMM_STATES}
       LSTM Lookback: {LSTM_LOOKBACK} days
       Anomaly Detection: {'Enabled' if ENABLE_ANOMALY_DETECTION else 'Disabled'}
    
    🤖 RL AGENT (SAC):
       Action Range: [{RL_ACTION_MIN}, {RL_ACTION_MAX}]
       Learning Rate: {RL_LEARNING_RATE}
       Batch Size: {RL_BATCH_SIZE}
    
    💰 RISK MANAGEMENT:
       Risk Per Trade: {RISK_PER_TRADE*100:.1f}%
       Max Gross Exposure: {MAX_GROSS_EXPOSURE*100:.0f}%
       Transaction Cost: {TRANSACTION_COST_BPS:.1f} bps
    
    📈 BACKTESTING:
       Walk-Forward Train: {WALK_FORWARD_TRAIN_YEARS} years
       Walk-Forward Test: {WALK_FORWARD_TEST_MONTHS} months
       Monte Carlo Paths: {MONTE_CARLO_PATHS:,}
    
    💵 INITIAL CAPITAL: ${INITIAL_CAPITAL:,.0f}
    """
    return summary
