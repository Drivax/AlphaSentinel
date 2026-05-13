"""
AlphaSentinel Signal Generator & Portfolio Manager
Translates regime + RL output into trading signals and manages portfolio risk.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import config
from src.utils import logger
from src.rl_agent import generate_signal, RLAgent

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """
    Generates trading signals combining regime detection and RL agent.
    """
    
    def __init__(self):
        """Initialize signal generator."""
        self.last_signal = None
        self.last_signal_date = None
        self.signal_history = []
        self.confidence_threshold = config.MIN_CONFIDENCE_TO_TRADE
        
    def generate_signal(
        self,
        market_state: np.ndarray,
        rl_agent: RLAgent,
        regime_info: Dict
    ) -> Dict:
        """
        Generate trading signal for current market state.
        
        Args:
            market_state: Current market state vector
            rl_agent: Trained RL agent
            regime_info: Regime detection output
            
        Returns:
            Signal dictionary with LONG/SHORT/NEUTRAL recommendation
        """
        signal_output = generate_signal(
            market_state,
            rl_agent,
            regime_info,
            self.confidence_threshold
        )
        
        self.last_signal = signal_output
        self.last_signal_date = datetime.now()
        self.signal_history.append(signal_output)
        
        return signal_output
    
    def get_signal_summary(self) -> str:
        """Return human-readable signal summary."""
        if self.last_signal is None:
            return "No signal generated yet"
        
        sig = self.last_signal
        summary = f"""
╔════════════════════════════════════════════════════╗
║              TRADING SIGNAL SUMMARY                ║
╠════════════════════════════════════════════════════╣
║ Signal:           {sig['signal']:<30}
║ Confidence:       {sig['confidence']*100:>7.1f}%            
║ Regime:           {sig['regime_name']:<30}
║ Position:         {sig['position']:>7.3f}            
║ Regime Multiplier:{sig['regime_multiplier']:>7.2f}x           
║ Anomaly Flag:     {'YES' if sig['is_anomaly'] else 'NO':<30}
╚════════════════════════════════════════════════════╝
        """
        return summary


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """
    Manages portfolio risk, position sizing, and risk controls.
    """
    
    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: position_size}
        self.entry_prices = {}  # {symbol: entry_price}
        self.entry_dates = {}  # {symbol: entry_date}
        
        self.trade_log = []
        self.daily_pnl_log = []
        self.daily_max_loss = 0.0
        self.halt_trading = False
        self.halt_until = None
        
    def update_price(self, symbol: str, current_price: float) -> Dict:
        """
        Update position with current market price.
        
        Returns position metrics (unrealized PnL, etc.)
        """
        try:
            if not symbol or current_price is None or current_price <= 0:
                logger.warning(f"Invalid price update: {symbol}={current_price}")
                return {'unrealized_pnl': 0.0, 'unrealized_pnl_pct': 0.0}
            
            if symbol not in self.positions or self.positions[symbol] == 0:
                return {'unrealized_pnl': 0.0, 'unrealized_pnl_pct': 0.0}
            
            entry_price = self.entry_prices.get(symbol, current_price)
            position_size = self.positions[symbol]
            
            if entry_price <= 0:
                logger.warning(f"Invalid entry price for {symbol}: {entry_price}")
                return {'unrealized_pnl': 0.0, 'unrealized_pnl_pct': 0.0}
            
            # Unrealized P&L
            unrealized_pnl = position_size * (current_price - entry_price)
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            # Check hard stop-loss
            if unrealized_pnl_pct < config.DAILY_STOP_LOSS:
                logger.warning(f"Hard stop triggered for {symbol}: {unrealized_pnl_pct*100:.2f}%")
                self.close_position(symbol, current_price)
            
            return {
                'unrealized_pnl': float(unrealized_pnl),
                'unrealized_pnl_pct': float(unrealized_pnl_pct),
            }
        except Exception as e:
            logger.error(f"Price update failed for {symbol}: {e}")
            return {'unrealized_pnl': 0.0, 'unrealized_pnl_pct': 0.0}
    
    def calculate_position_size(
        self,
        signal: str,
        volatility: float,
        regime: int
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Formula: Position = Risk_Budget / (Volatility * Regime_Adjustment)
        
        Args:
            signal: LONG, SHORT, or NEUTRAL
            volatility: Current realized volatility
            regime: Current regime (0, 1, or 2)
            
        Returns:
            Position size (-1.0 to +1.0)
        """
        try:
            if signal == config.SIGNAL_NEUTRAL:
                return 0.0
            
            # Validate inputs
            if volatility is None or volatility <= 0:
                logger.warning(f"Invalid volatility: {volatility}, using baseline")
                volatility = config.BASELINE_VOLATILITY
            
            if not np.isfinite(volatility):
                logger.warning("Non-finite volatility, using baseline")
                volatility = config.BASELINE_VOLATILITY
            
            # Baseline risk per trade
            risk_budget = config.RISK_PER_TRADE
            
            # Normalize volatility
            vol_adjusted = volatility / max(config.BASELINE_VOLATILITY, 0.001)
            
            # Regime multiplier (reduce in volatile/crisis)
            regime_mult = config.REGIME_POSITION_MULTIPLIERS.get(regime, 1.0)
            
            # Calculate position
            position_base = risk_budget / vol_adjusted * regime_mult
            
            # Clip to action space
            position = np.clip(position_base, -1.0, 1.0)
            
            # Direction
            if signal == config.SIGNAL_SHORT:
                position = -abs(position)
            else:  # LONG
                position = abs(position)
            
            return float(position)
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def check_portfolio_limits(self, proposed_position: float) -> bool:
        """
        Check if proposed position violates portfolio limits.
        
        Returns True if position is acceptable.
        """
        try:
            if not np.isfinite(proposed_position):
                logger.warning("Non-finite proposed position")
                return False
            
            # Total gross exposure
            gross_sum = sum(abs(p) for p in self.positions.values() if np.isfinite(p))
            total_gross = gross_sum + abs(proposed_position)
            
            if total_gross > config.MAX_GROSS_EXPOSURE:
                logger.warning(f"Gross exposure limit exceeded: {total_gross:.2f} > {config.MAX_GROSS_EXPOSURE}")
                return False
            
            # Net exposure
            net_sum = sum(p for p in self.positions.values() if np.isfinite(p))
            total_net = net_sum + proposed_position
            
            if abs(total_net) > config.MAX_NET_EXPOSURE:
                logger.warning(f"Net exposure limit exceeded: {abs(total_net):.2f} > {config.MAX_NET_EXPOSURE}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Portfolio limit check failed: {e}")
            return False
    
    def open_position(
        self,
        symbol: str,
        signal: str,
        position_size: float,
        entry_price: float,
        timestamp: datetime
    ) -> bool:
        """
        Open new trading position.
        
        Returns True if position opened successfully.
        """
        # Check if already in position
        if symbol in self.positions and self.positions[symbol] != 0:
            logger.warning(f"Already in position for {symbol}. Close first.")
            return False
        
        # Check portfolio limits
        if not self.check_portfolio_limits(position_size):
            return False
        
        # Check daily loss limit
        if self.halt_trading and datetime.now() < self.halt_until:
            logger.warning("Trading halted due to daily loss limit")
            return False
        
        # Open position
        self.positions[symbol] = position_size
        self.entry_prices[symbol] = entry_price
        self.entry_dates[symbol] = timestamp
        
        logger.info(f"Opened {signal} position: {symbol} x {position_size:.2f} @ {entry_price:.2f}")
        
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """
        Close existing position and record trade.
        
        Returns trade P&L information.
        """
        if symbol not in self.positions or self.positions[symbol] == 0:
            return {}
        
        position_size = self.positions[symbol]
        entry_price = self.entry_prices[symbol]
        entry_date = self.entry_dates[symbol]
        
        # Calculate P&L
        pnl = position_size * (exit_price - entry_price)
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # Transaction costs
        txn_cost = abs(position_size) * config.TRANSACTION_COST_BPS / 10000
        net_pnl = pnl - txn_cost
        
        # Update capital
        self.current_capital += net_pnl
        daily_loss = net_pnl / self.initial_capital
        
        # Log trade
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': datetime.now(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'gross_pnl': pnl,
            'txn_cost': txn_cost,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
        }
        self.trade_log.append(trade)
        
        # Check daily loss limit
        self.daily_max_loss = min(self.daily_max_loss, daily_loss)
        if self.daily_max_loss < -config.MAX_DAILY_LOSS:
            logger.warning(f"Daily loss limit hit: {self.daily_max_loss*100:.2f}% < {-config.MAX_DAILY_LOSS*100:.2f}%")
            self.halt_trading = True
            self.halt_until = datetime.now() + timedelta(hours=24)
        
        # Close position
        self.positions[symbol] = 0.0
        
        logger.info(f"Closed position: {symbol} | PnL: {net_pnl:.2f} ({pnl_pct*100:+.2f}%)")
        
        return trade
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        total_value = self.current_capital
        
        for symbol, position_size in self.positions.items():
            if symbol in current_prices and position_size != 0:
                current_price = current_prices[symbol]
                entry_price = self.entry_prices.get(symbol, current_price)
                unrealized_pnl = position_size * (current_price - entry_price)
                total_value += unrealized_pnl
        
        return total_value
    
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Get current portfolio metrics."""
        total_value = self.get_portfolio_value(current_prices)
        
        total_gross_exposure = sum(abs(p) for p in self.positions.values())
        total_net_exposure = sum(self.positions.values())
        
        return {
            'total_value': total_value,
            'capital': self.current_capital,
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'total_return_pct': (total_value - self.initial_capital) / self.initial_capital * 100,
            'gross_exposure': total_gross_exposure,
            'net_exposure': total_net_exposure,
            'num_open_positions': sum(1 for p in self.positions.values() if p != 0),
            'daily_max_loss': self.daily_max_loss,
            'trading_halted': self.halt_trading,
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trade log to DataFrame."""
        if len(self.trade_log) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_log)
        return df
    
    def reset_daily_limits(self):
        """Reset daily loss tracking (call at start of new trading day)."""
        self.daily_max_loss = 0.0
        
        # Check if halt period has expired
        if self.halt_trading and datetime.now() > self.halt_until:
            self.halt_trading = False
            logger.info("Trading resumed after halt period")
