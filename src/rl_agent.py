"""
AlphaSentinel RL Agent Module
Soft Actor-Critic (SAC) agent for adaptive signal generation.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

import config
from src.utils import logger

# ============================================================================
# TRADING ENVIRONMENT
# ============================================================================

class TradingEnvironment(gym.Env):
    """
    OpenAI Gym environment for RL training.
    
    State: [returns, vol, skew, regime_vector, portfolio_delta, cash_level]
    Action: Position change (-1 to +1)
    Reward: Sharpe - MaxDD - TxnCost - Turnover
    """
    
    def __init__(
        self,
        price_data: np.ndarray,
        regime_data: np.ndarray,
        lookback: int = 20
    ):
        """
        Initialize trading environment.
        
        Args:
            price_data: Daily prices (n_days,)
            regime_data: Regime probabilities (n_days, n_regimes)
            lookback: Days for metrics calculation
        """
        self.price_data = price_data
        self.regime_data = regime_data
        self.lookback = lookback
        
        self.current_idx = lookback
        self.position = 0.0  # Current position (-1 to +1)
        self.equity = 1.0  # Normalized equity
        self.equity_history = [1.0]
        self.peak_equity = 1.0
        
        # Compute features
        self.returns = np.diff(np.log(price_data))
        self.volatility = self._compute_rolling_vol()
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([config.RL_ACTION_MIN]),
            high=np.array([config.RL_ACTION_MAX]),
            dtype=np.float32
        )
        
        # State: [return, vol, regime_prob_0, regime_prob_1, regime_prob_2, pos, cash]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )
    
    def _compute_rolling_vol(self) -> np.ndarray:
        """Compute 20-day rolling volatility."""
        vol = np.zeros(len(self.returns))
        for i in range(self.lookback, len(self.returns)):
            vol[i] = np.std(self.returns[i - self.lookback:i])
        return vol
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        idx = self.current_idx
        
        # Price return
        ret = float(self.returns[idx])
        
        # Volatility
        vol = float(self.volatility[idx])
        
        # Regime probabilities
        regime_probs = self.regime_data[idx]
        
        # Position and cash
        pos = float(self.position)
        cash = 1.0 - abs(self.position)
        
        state = np.array([ret, vol, *regime_probs, pos, cash], dtype=np.float32)
        return state
    
    def _compute_reward(self, old_position: float, new_position: float) -> float:
        """
        Compute reward for action.
        
        Reward = alpha1*Sharpe - alpha2*MaxDD - alpha3*TxnCost - alpha4*Turnover
        """
        # Daily return from position
        idx = self.current_idx
        price_return = self.returns[idx]
        daily_pnl = self.position * price_return
        
        # Update equity
        transaction_cost = abs(new_position - old_position) * config.TRANSACTION_COST_BPS / 10000
        self.equity *= (1 + daily_pnl - transaction_cost)
        self.equity_history.append(self.equity)
        
        # Compute metrics
        equity_arr = np.array(self.equity_history[-self.lookback:])
        returns_arr = np.diff(np.log(equity_arr))
        
        # Sharpe ratio
        if len(returns_arr) > 1:
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = np.max(equity_arr)
        drawdown = (np.min(equity_arr) - peak) / peak if peak > 0 else 0.0
        
        # Reward components
        reward = (
            config.REWARD_ALPHA_SHARPE * sharpe -
            config.REWARD_ALPHA_MAXDD * abs(drawdown) -
            config.REWARD_ALPHA_TXNCOST * transaction_cost -
            config.REWARD_ALPHA_TURNOVER * abs(new_position - old_position)
        )
        
        return float(reward)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in environment."""
        new_position = float(np.clip(action[0], -1, 1))
        reward = self._compute_reward(self.position, new_position)
        self.position = new_position
        
        self.current_idx += 1
        done = self.current_idx >= len(self.price_data) - 1
        
        state = self._get_state() if not done else np.zeros(7, dtype=np.float32)
        
        return state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_idx = self.lookback
        self.position = 0.0
        self.equity = 1.0
        self.equity_history = [1.0]
        self.peak_equity = 1.0
        
        return self._get_state()
    
    def render(self, mode='human'):
        """Render environment state."""
        print(f"Step {self.current_idx}, Position: {self.position:.2f}, Equity: {self.equity:.4f}")


# ============================================================================
# RL AGENT
# ============================================================================

class RLAgent:
    """
    Soft Actor-Critic agent for adaptive trading signal generation.
    """
    
    def __init__(self):
        """Initialize RL agent."""
        self.agent = None
        self.is_trained = False
        self.training_history = []
    
    def train(
        self,
        env: TradingEnvironment,
        total_timesteps: int = config.RL_TOTAL_TIMESTEPS,
        learning_rate: float = config.RL_LEARNING_RATE
    ):
        """
        Train SAC agent.
        
        Args:
            env: Training environment
            total_timesteps: Total training steps
            learning_rate: SAC learning rate
        """
        logger.info(f"Training SAC agent for {total_timesteps:,} timesteps...")
        
        try:
            # Create SAC agent
            self.agent = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                gamma=config.RL_GAMMA,
                tau=config.RL_TAU,
                batch_size=config.RL_BATCH_SIZE,
                buffer_size=config.RL_BUFFER_SIZE,
                ent_coef='auto',
                verbose=0  # Reduced verbosity
            )
            
            # Train with error handling
            self.agent.learn(total_timesteps=total_timesteps)
            self.is_trained = True
            logger.info("[OK] SAC agent training complete")
        except Exception as e:
            logger.error(f"SAC training failed: {e}")
            logger.warning("Continuing without trained RL agent")
            self.is_trained = False
    
    def predict(self, state: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Get action for given state.
        
        Args:
            state: State vector
            
        Returns:
            Tuple of (action, info)
        """
        if state is None or not np.all(np.isfinite(state)):
            # Invalid state - return neutral action
            return np.array([0.0]), None
        
        if not self.is_trained or self.agent is None:
            # No trained agent - return neutral action
            return np.array([0.0]), None
        
        try:
            action, info = self.agent.predict(state, deterministic=True)
            # Ensure action is in valid range
            action = np.clip(action, config.RL_ACTION_MIN, config.RL_ACTION_MAX)
            return action, info
        except Exception as e:
            logger.debug(f"RL prediction failed: {e}")
            return np.array([0.0]), None
    
    def save(self, path: str = config.RL_MODEL_PATH):
        """Save trained agent."""
        if self.agent is not None:
            self.agent.save(path)
            logger.info(f"[OK] RL agent saved to {path}")
    
    def load(self, path: str = config.RL_MODEL_PATH):
        """Load pre-trained agent."""
        try:
            self.agent = SAC.load(path)
            self.is_trained = True
            logger.info(f"[OK] RL agent loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")


# ============================================================================
# SIGNAL GENERATION FROM RL OUTPUT
# ============================================================================

def position_to_signal(position: float) -> str:
    """
    Convert continuous position to discrete signal.
    
    Args:
        position: Position value (-1 to +1)
        
    Returns:
        Signal: "LONG", "SHORT", or "NEUTRAL"
    """
    if position > config.RL_ACTION_THRESHOLD_LONG:
        return config.SIGNAL_LONG
    elif position < config.RL_ACTION_THRESHOLD_SHORT:
        return config.SIGNAL_SHORT
    else:
        return config.SIGNAL_NEUTRAL


def generate_signal(
    state: np.ndarray,
    rl_agent: RLAgent,
    regime_info: Dict,
    confidence_threshold: float = config.MIN_CONFIDENCE_TO_TRADE
) -> Dict:
    """
    Generate trading signal from market state using RL agent.
    
    Args:
        state: Current market state
        rl_agent: Trained RL agent
        regime_info: Regime detection output
        confidence_threshold: Minimum confidence to generate signal
        
    Returns:
        Dict with signal and metadata
    """
    try:
        # Validate inputs
        if state is None or not np.all(np.isfinite(state)):
            logger.warning("Invalid state for signal generation")
            return {
                'signal': config.SIGNAL_NEUTRAL,
                'position': 0.0,
                'confidence': 0.0,
                'regime': regime_info.get('regime', 0),
                'regime_name': regime_info.get('regime_name', 'Unknown'),
                'regime_multiplier': 1.0,
                'is_anomaly': regime_info.get('is_anomaly', False),
            }
        
        # Get RL action
        action, _ = rl_agent.predict(state)
        position = float(action[0]) if action is not None else 0.0
        position = np.clip(position, -1.0, 1.0)
        
        # Convert to discrete signal
        signal = position_to_signal(position)
        
        # Compute confidence
        regime_confidence = regime_info.get('confidence', 0.5)
        regime = regime_info.get('regime', 0)
        
        # Apply regime-dependent sizing
        regime_multiplier = config.REGIME_POSITION_MULTIPLIERS.get(regime, 1.0)
        
        # Adjust confidence based on regime uncertainty
        if regime_info.get('is_anomaly', False):
            confidence = regime_confidence * 0.7  # Reduce confidence on anomaly
        else:
            confidence = regime_confidence
        
        # Generate signal only if confident enough
        if confidence < confidence_threshold:
            signal = config.SIGNAL_NEUTRAL
        
        return {
            'signal': signal,
            'position': position,
            'confidence': float(confidence),
            'regime': regime,
            'regime_name': regime_info.get('regime_name', 'Unknown'),
            'regime_multiplier': regime_multiplier,
            'is_anomaly': regime_info.get('is_anomaly', False),
        }
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        return {
            'signal': config.SIGNAL_NEUTRAL,
            'position': 0.0,
            'confidence': 0.0,
            'regime': regime_info.get('regime', 0),
            'regime_name': regime_info.get('regime_name', 'Unknown'),
            'regime_multiplier': 1.0,
            'is_anomaly': regime_info.get('is_anomaly', False),
        }
