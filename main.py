"""
AlphaSentinel Main Entry Point
Orchestrates regime detection, signal generation, and backtesting pipeline.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import config
from src.utils import logger, format_metrics_table, generate_performance_report
from src.data_loader import (
    load_price_data,
    align_multiasset_data,
    compute_equal_weight_index,
    extract_regime_features,
    create_walk_forward_splits
)
from src.regime_detector import RegimeDetector
from src.rl_agent import RLAgent, TradingEnvironment
from src.signal_generator import SignalGenerator, PortfolioManager
from src.backtester import Backtester

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_banner():
    """Print AlphaSentinel banner."""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║            🚀  ALPHA SENTINEL - REGIME DETECTION  🚀          ║
    ║                                                                ║
    ║    Professional Quantitative Trading System                   ║
    ║    HMM + LSTM + Isolation Forest + SAC RL Agent               ║
    ║                                                                ║
    ║    Target Markets: CAC40, DAX, Euro Stoxx 50                  ║
    ║    Deployment: Real-time Streamlit Dashboard                  ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config_summary():
    """Print configuration summary."""
    print(config.get_config_summary())


# ============================================================================
# BACKTEST PIPELINE
# ============================================================================

def run_backtest(args):
    """Run full backtesting pipeline."""
    print_banner()
    print_config_summary()
    
    logger.info("=" * 70)
    logger.info("STARTING FULL BACKTEST PIPELINE")
    logger.info("=" * 70)
    
    try:
        # 1. Load data
        logger.info("\n[1/5] Loading market data...")
        data = load_price_data(
            symbols=config.SYMBOLS,
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=True
        )
        
        if not data or len(data) == 0:
            logger.error("Failed to load price data")
            return
        
        # Align multi-asset data and compute portfolio
        aligned_prices = align_multiasset_data(data)
        portfolio_value = compute_equal_weight_index(aligned_prices)
        
        logger.info(f"✓ Loaded {len(portfolio_value)} trading days")
        
        # 2. Train regime detector
        logger.info("\n[2/5] Training regime detection ensemble...")
        features = extract_regime_features(
            pd.DataFrame({'Close': portfolio_value}),
            lookback=config.VOLATILITY_WINDOW
        )
        
        if len(features) < config.HMM_STATES + 5:
            logger.error(f"Insufficient data for regime detection: {len(features)} rows")
            return
        
        regime_detector = RegimeDetector()
        try:
            regime_detector.fit(features)
            regime_detector.save()
            logger.info("✓ Regime detector trained and saved")
        except Exception as e:
            logger.error(f"Regime detector training failed: {e}")
            return
        
        # 3. Predict regimes on full data
        logger.info("\n[3/5] Predicting regimes for full period...")
        try:
            regime_predictions = regime_detector.predict_timeseries(features)
            logger.info(f"✓ Predicted regimes for {len(regime_predictions)} dates")
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return
        
        # 4. Backtesting
        logger.info("\n[4/5] Running walk-forward backtest...")
        
        backtester = Backtester()
        
        # Convert to trading days (252 per year)
        train_periods = int(config.WALK_FORWARD_TRAIN_YEARS * 252)
        test_periods = int(config.WALK_FORWARD_TEST_MONTHS / 12 * 252)
        
        # Simple strategy: buy in calm regimes, neutral otherwise
        def simple_strategy(regime, price):
            if regime == 0:  # Calm trending
                return "LONG"
            elif regime == 1:  # Volatile
                return "NEUTRAL"
            else:  # Crisis
                return "NEUTRAL"
        
        try:
            fold_results = backtester.walk_forward_backtest(
                strategy_func=simple_strategy,
                price_data=portfolio_value,
                regime_data=regime_predictions,
                train_periods=train_periods,
                test_periods=test_periods,
                step_periods=test_periods  # Non-overlapping folds
            )
            
            if not fold_results:
                logger.warning("No fold results generated")
            else:
                logger.info(f"✓ Walk-forward backtest complete ({len(fold_results)} folds)")
        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            return
        
        # 5. Monte Carlo simulation
        logger.info("\n[5/5] Running Monte Carlo simulation...")
        
        try:
            if backtester.trades:
                mc_results = backtester.monte_carlo_simulation(
                    trades=backtester.trades,
                    num_paths=config.MONTE_CARLO_PATHS
                )
                logger.info("✓ Monte Carlo simulation complete")
            else:
                logger.warning("No trades to simulate")
        except Exception as e:
            logger.warning(f"Monte Carlo simulation failed: {e}")
        
        # 6. Print results
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        print(backtester.get_summary_report())
        
        # Save results
        try:
            backtester.save_results()
            backtester.save_trades()
            logger.info("✓ Results saved")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        logger.info("\n✓ BACKTEST PIPELINE COMPLETE")
    
    except Exception as e:
        logger.error(f"Backtest pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================================================
# PAPER TRADING / LIVE SIMULATION
# ============================================================================

def run_paper_trade(args):
    """Run paper trading (latest signal generation)."""
    print_banner()
    
    logger.info("=" * 70)
    logger.info("RUNNING PAPER TRADING MODE (Latest Signal)")
    logger.info("=" * 70)
    
    # Load latest data
    logger.info("\nLoading latest market data...")
    data = load_price_data(use_cache=False)
    
    if not data:
        logger.error("Failed to load data")
        return
    
    # Prepare data
    aligned_prices = align_multiasset_data(data)
    portfolio_value = compute_equal_weight_index(aligned_prices)
    
    # Extract features
    features = extract_regime_features(
        pd.DataFrame({'Close': portfolio_value}),
        lookback=config.VOLATILITY_WINDOW
    )
    
    # Load or train regime detector
    logger.info("Loading regime detector...")
    regime_detector = RegimeDetector()
    
    try:
        regime_detector.load()
    except:
        logger.warning("Pre-trained detector not found. Training fresh...")
        regime_detector.fit(features)
    
    # Get latest regime
    logger.info("Detecting current regime...")
    latest_features = features.tail(config.LSTM_LOOKBACK)
    regime_info = regime_detector.predict(latest_features)
    
    # Generate signal
    logger.info("Generating trading signal...")
    signal_gen = SignalGenerator()
    
    # Create state vector for RL (simplified)
    state = np.array([
        float(features['returns'].iloc[-1]),
        float(features['volatility'].iloc[-1]),
        float(features['skewness'].iloc[-1]),
        *regime_info['probabilities'],
        0.0,  # portfolio_delta
        1.0   # cash_level
    ], dtype=np.float32)
    
    # Get RL agent
    rl_agent = RLAgent()
    try:
        rl_agent.load()
    except:
        logger.warning("RL agent not found. Using random action.")
    
    # Generate signal
    signal = signal_gen.generate_signal(state, rl_agent, regime_info)
    
    # Print results
    print(signal_gen.get_signal_summary())
    
    logger.info(f"Regime Probabilities: {regime_info['probabilities']}")
    logger.info(f"Anomaly Detected: {regime_info['is_anomaly']}")
    logger.info(f"Signal Confidence: {signal['confidence']*100:.1f}%")
    
    logger.info("\n✓ PAPER TRADING COMPLETE")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AlphaSentinel: Regime Detection & Trading System"
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run full backtesting pipeline'
    )
    
    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Generate latest trading signal (paper trading)'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch Streamlit dashboard'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=config.START_DATE,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=config.END_DATE,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=config.SYMBOLS,
        help='Market symbols to trade'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging output'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help and run backtest by default
    if len(sys.argv) == 1:
        print_banner()
        print("\nUsage: python main.py [OPTIONS]")
        print("\nExamples:")
        print("  python main.py --backtest")
        print("  python main.py --paper-trade")
        print("  python main.py --dashboard")
        print("\nFor full help: python main.py --help")
        
        # Run backtest by default
        print("\n🔄 Running default backtest pipeline...\n")
        run_backtest(args)
        return
    
    # Execute requested command
    if args.backtest:
        run_backtest(args)
    
    elif args.paper_trade:
        run_paper_trade(args)
    
    elif args.dashboard:
        print_banner()
        logger.info("Launching Streamlit dashboard...")
        logger.info("Open http://localhost:8501 in your browser")
        import subprocess
        subprocess.run(['streamlit', 'run', 'app.py'])
    
    else:
        # Default to backtest
        run_backtest(args)


if __name__ == '__main__':
    main()
