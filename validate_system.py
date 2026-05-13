"""
Validation script to verify all components work correctly with error handling.
Tests data loading, regime detection, signal generation, and backtesting.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_imports():
    """Verify all dependencies can be imported."""
    logger.info("\n" + "="*70)
    logger.info("VALIDATING IMPORTS")
    logger.info("="*70)
    
    required_modules = [
        'numpy', 'pandas', 'tensorflow', 'sklearn',
        'hmmlearn', 'stable_baselines3', 'gym', 'yfinance'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"\nMissing dependencies: {', '.join(failed_imports)}")
        return False
    
    logger.info("\n✓ All dependencies available")
    return True

def validate_config():
    """Verify configuration loads without errors."""
    logger.info("\n" + "="*70)
    logger.info("VALIDATING CONFIG")
    logger.info("="*70)
    
    try:
        from config import *
        
        # Validate key parameters exist
        assert SYMBOLS, "SYMBOLS not defined"
        assert START_DATE, "START_DATE not defined"
        assert END_DATE, "END_DATE not defined"
        assert INITIAL_CAPITAL > 0, "INITIAL_CAPITAL must be positive"
        assert HMM_STATES > 0, "HMM_STATES must be positive"
        assert LSTM_LOOKBACK > 0, "LSTM_LOOKBACK must be positive"
        
        logger.info(f"✓ Symbols: {SYMBOLS}")
        logger.info(f"✓ Period: {START_DATE} to {END_DATE}")
        logger.info(f"✓ Initial capital: ${INITIAL_CAPITAL:,.0f}")
        logger.info(f"✓ HMM states: {HMM_STATES}")
        logger.info(f"✓ LSTM lookback: {LSTM_LOOKBACK}")
        
        logger.info("\n✓ Config valid")
        return True
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False

def validate_data_loading():
    """Test data loading with error handling."""
    logger.info("\n" + "="*70)
    logger.info("VALIDATING DATA LOADING")
    logger.info("="*70)
    
    try:
        from src.data_loader import load_price_data, align_multiasset_data, compute_equal_weight_index
        from config import SYMBOLS, START_DATE, END_DATE
        
        # Try loading data
        logger.info("Loading price data...")
        data = load_price_data(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            use_cache=True
        )
        
        if not data:
            logger.error("No data loaded")
            return False
        
        logger.info(f"✓ Loaded {len(data[list(data.keys())[0]])} days for {len(data)} symbols")
        
        # Test alignment
        logger.info("Aligning multi-asset data...")
        aligned = align_multiasset_data(data)
        logger.info(f"✓ Aligned: {len(aligned)} days")
        
        # Test index computation
        logger.info("Computing portfolio index...")
        portfolio = compute_equal_weight_index(aligned)
        logger.info(f"✓ Portfolio index: {len(portfolio)} days")
        
        logger.info("\n✓ Data loading validated")
        return True
    except Exception as e:
        logger.error(f"Data loading validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def validate_regime_detection():
    """Test regime detection with error handling."""
    logger.info("\n" + "="*70)
    logger.info("VALIDATING REGIME DETECTION")
    logger.info("="*70)
    
    try:
        from src.data_loader import (
            load_price_data, align_multiasset_data, 
            compute_equal_weight_index, extract_regime_features
        )
        from src.regime_detector import RegimeDetector
        from config import SYMBOLS, START_DATE, END_DATE, VOLATILITY_WINDOW
        import pandas as pd
        
        # Load and prepare data
        logger.info("Loading data for regime detection...")
        data = load_price_data(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            use_cache=True
        )
        
        if not data:
            logger.error("No data loaded")
            return False
        
        aligned = align_multiasset_data(data)
        portfolio = compute_equal_weight_index(aligned)
        
        # Extract features
        logger.info("Extracting regime features...")
        features = extract_regime_features(
            pd.DataFrame({'Close': portfolio}),
            lookback=VOLATILITY_WINDOW
        )
        
        if len(features) < 100:
            logger.error(f"Insufficient features: {len(features)}")
            return False
        
        logger.info(f"✓ Extracted {len(features)} feature vectors")
        
        # Train regime detector
        logger.info("Training regime detector...")
        detector = RegimeDetector()
        detector.fit(features)
        logger.info("✓ Regime detector trained")
        
        # Test prediction
        logger.info("Testing regime prediction...")
        pred = detector.predict(features.iloc[-60:])
        logger.info(f"✓ Prediction: Regime {pred['regime']} ({pred['regime_name']})")
        logger.info(f"  Confidence: {pred['confidence']:.2%}")
        
        logger.info("\n✓ Regime detection validated")
        return True
    except Exception as e:
        logger.error(f"Regime detection validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def validate_signal_generation():
    """Test signal generation with error handling."""
    logger.info("\n" + "="*70)
    logger.info("VALIDATING SIGNAL GENERATION")
    logger.info("="*70)
    
    try:
        from src.signal_generator import generate_signal, PortfolioManager
        from src.rl_agent import RLAgent
        import numpy as np
        
        # Create mock regime info
        regime_info = {
            'regime': 0,
            'regime_name': 'Calm',
            'confidence': 0.85,
            'is_anomaly': False
        }
        
        # Create RL agent
        logger.info("Initializing RL agent...")
        rl_agent = RLAgent()
        logger.info("✓ RL agent ready")
        
        # Test signal generation
        logger.info("Testing signal generation...")
        state = np.random.randn(5)  # Mock state
        signal = generate_signal(state, rl_agent, regime_info)
        logger.info(f"✓ Signal generated: {signal['signal']}")
        
        # Test portfolio manager
        logger.info("Testing portfolio manager...")
        pm = PortfolioManager(initial_capital=100000)
        logger.info(f"✓ Portfolio manager initialized with ${pm.current_capital:,.0f}")
        
        # Test position sizing
        position_size = pm.calculate_position_size(
            signal='LONG',
            volatility=0.02,
            regime=0
        )
        logger.info(f"✓ Position size calculated: {position_size:.2%}")
        
        logger.info("\n✓ Signal generation validated")
        return True
    except Exception as e:
        logger.error(f"Signal generation validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all validations."""
    logger.info("\n")
    logger.info("█" * 70)
    logger.info("ALPHASENTINEL SYSTEM VALIDATION")
    logger.info("█" * 70)
    
    validations = [
        ("Imports", validate_imports),
        ("Configuration", validate_config),
        ("Data Loading", validate_data_loading),
        ("Regime Detection", validate_regime_detection),
        ("Signal Generation", validate_signal_generation),
    ]
    
    results = {}
    for name, validator in validations:
        try:
            results[name] = validator()
        except Exception as e:
            logger.error(f"Validation {name} crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} - {name}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} validations passed")
    
    if total_passed == total_tests:
        logger.info("\n✓ All systems operational!")
        return 0
    else:
        logger.error(f"\n✗ {total_tests - total_passed} validation(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
