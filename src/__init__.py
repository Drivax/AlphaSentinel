"""
AlphaSentinel - Professional Regime Detection & Trading System

A complete quantitative trading system that detects market regime shifts
in real-time and generates actionable trading signals using:
- Hidden Markov Models (HMM)
- LSTM neural networks
- Isolation Forest anomaly detection
- Soft Actor-Critic (SAC) reinforcement learning

Target Markets: CAC40, DAX, Euro Stoxx 50
Deployment: Real-time Streamlit dashboard + professional backtesting framework

Author: Quantitative Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Drivax"
__all__ = [
    'data_loader',
    'regime_detector',
    'rl_agent',
    'signal_generator',
    'backtester',
    'utils',
]
