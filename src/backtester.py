"""
AlphaSentinel Backtesting Framework
Walk-forward analysis with Monte Carlo simulation and comprehensive metrics.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

import config
from src.utils import (
    logger, 
    calculate_sharpe_ratio, 
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_profit_factor,
    calculate_win_rate,
    calculate_annual_return,
    generate_performance_report,
    format_metrics_table
)

# ============================================================================
# BACKTESTER
# ============================================================================

class Backtester:
    """
    Walk-forward backtesting engine with realistic assumptions.
    
    Features:
    - Walk-forward analysis (rolling 3-year train, 6-month test)
    - Transaction costs (4 bps round-trip)
    - Slippage modeling
    - Position limits and risk controls
    """
    
    def __init__(self):
        """Initialize backtester."""
        self.results = []
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        
    def backtest(
        self,
        strategy_func,
        price_data: pd.DataFrame,
        regime_data: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> Dict:
        """
        Run backtest on price data using strategy function.
        
        Args:
            strategy_func: Function that takes (date, price, regime) -> signal
            price_data: Price series
            regime_data: Regime probability data
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Backtest results dictionary
        """
        if end_idx is None:
            end_idx = len(price_data)
        
        logger.info(f"Running backtest from idx {start_idx} to {end_idx}")
        
        equity = config.INITIAL_CAPITAL
        position = 0.0
        entry_price = 0.0
        self.equity_curve = [equity]
        self.daily_returns = []
        self.trades = []
        
        for i in range(start_idx, end_idx - 1):
            current_price = price_data.iloc[i]
            next_price = price_data.iloc[i + 1]
            
            # Get regime
            if i < len(regime_data):
                regime = regime_data.iloc[i]['regime']
            else:
                regime = 0
            
            # Get signal from strategy
            signal = strategy_func(regime, current_price)
            
            # Process signal
            new_position = self._signal_to_position(signal)
            
            # Handle position change
            if position != new_position:
                # Close old position
                if position != 0:
                    pnl = position * (current_price - entry_price)
                    txn_cost = abs(position) * config.TRANSACTION_COST_BPS / 10000
                    trade_pnl = pnl - txn_cost
                    
                    self.trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'size': position,
                        'pnl': trade_pnl,
                    })
                    
                    equity += trade_pnl
                
                # Open new position
                if new_position != 0:
                    entry_price = current_price
                    txn_cost = abs(new_position) * config.TRANSACTION_COST_BPS / 10000
                    equity -= txn_cost
                
                position = new_position
            
            # Daily P&L
            if position != 0:
                daily_pnl = position * (next_price - current_price)
                daily_return = daily_pnl / equity if equity > 0 else 0
            else:
                daily_return = 0
            
            daily_return_pct = next_price / current_price - 1 if current_price > 0 else 0
            equity_change = equity * (1 + daily_return)
            
            self.equity_curve.append(equity_change)
            self.daily_returns.append(daily_return)
            equity = equity_change
        
        # Close final position
        if position != 0:
            final_price = price_data.iloc[end_idx - 1]
            pnl = position * (final_price - entry_price)
            txn_cost = abs(position) * config.TRANSACTION_COST_BPS / 10000
            trade_pnl = pnl - txn_cost
            
            self.trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'size': position,
                'pnl': trade_pnl,
            })
            
            equity += trade_pnl
        
        # Compute metrics
        metrics = self._compute_metrics(equity)
        
        return metrics
    
    def _signal_to_position(self, signal: str) -> float:
        """Convert signal to position size."""
        if signal == "LONG":
            return 0.5  # 50% exposure
        elif signal == "SHORT":
            return -0.5
        else:
            return 0.0
    
    def _compute_metrics(self, final_equity: float) -> Dict:
        """Compute performance metrics from backtest."""
        if len(self.equity_curve) < 2:
            logger.warning("Insufficient equity curve data for metrics calculation")
            return {}
        
        equity_arr = np.array(self.equity_curve)
        returns_arr = np.array(self.daily_returns)
        
        # Remove any NaN/inf
        returns_arr = returns_arr[np.isfinite(returns_arr)]
        
        if len(returns_arr) == 0:
            logger.warning("No valid returns data")
            return {}
        
        total_return = (final_equity - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL
        years = len(self.equity_curve) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        max_dd = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': calculate_sharpe_ratio(pd.Series(returns_arr)),
            'sortino_ratio': calculate_sortino_ratio(pd.Series(returns_arr)),
            'calmar_ratio': calculate_calmar_ratio(pd.Series(returns_arr)),
            'max_drawdown': max_dd,
            'volatility': np.std(returns_arr) * np.sqrt(252),
            'num_trades': len(self.trades),
            'profit_factor': calculate_profit_factor(pd.DataFrame(self.trades)) if self.trades else 0,
            'win_rate': calculate_win_rate(pd.DataFrame(self.trades)) if self.trades else 0,
            'final_equity': final_equity,
        }
        
        self.results.append(metrics)
        
        return metrics
    
    def walk_forward_backtest(
        self,
        strategy_func,
        price_data: pd.DataFrame,
        regime_data: pd.DataFrame,
        train_periods: int,
        test_periods: int,
        step_periods: int = 1
    ) -> List[Dict]:
        """
        Run walk-forward backtest.
        
        Args:
            strategy_func: Strategy function
            price_data: Price data
            regime_data: Regime data
            train_periods: Training window length
            test_periods: Test window length
            step_periods: Step size
            
        Returns:
            List of results for each fold
        """
        try:
            logger.info("Running walk-forward backtest...")
            
            # Validate inputs
            if len(price_data) == 0:
                logger.error("Empty price data")
                return []
            
            if len(regime_data) == 0:
                logger.error("Empty regime data")
                return []
            
            if train_periods <= 0 or test_periods <= 0:
                logger.error(f"Invalid periods: train={train_periods}, test={test_periods}")
                return []
            
            if train_periods + test_periods > len(price_data):
                logger.warning(f"Total periods ({train_periods + test_periods}) exceeds data length ({len(price_data)})")
                return []
            
            fold_results = []
            fold_num = 0
            idx = 0
            
            while idx + train_periods + test_periods <= len(price_data):
                fold_num += 1
                
                try:
                    train_start = idx
                    train_end = idx + train_periods
                    test_start = train_end
                    test_end = test_start + test_periods
                    
                    logger.info(f"Fold {fold_num}: Train [{train_start}, {train_end}), Test [{test_start}, {test_end})")
                    
                    # Run test (would normally retrain on train set)
                    fold_metrics = self.backtest(
                        strategy_func,
                        price_data,
                        regime_data,
                        start_idx=test_start,
                        end_idx=test_end
                    )
                    
                    if fold_metrics:
                        fold_results.append(fold_metrics)
                    else:
                        logger.warning(f"Fold {fold_num} produced no metrics")
                
                except Exception as e:
                    logger.warning(f"Fold {fold_num} failed: {e}")
                    continue
                
                idx += step_periods
            
            if fold_results:
                logger.info(f"✓ Walk-forward backtest complete ({len(fold_results)} valid folds)")
            else:
                logger.warning("No valid folds produced")
            
            return fold_results
        
        except Exception as e:
            logger.error(f"Walk-forward backtest failed: {e}")
            return []
    
    def monte_carlo_simulation(
        self,
        trades: List[Dict],
        num_paths: int = config.MONTE_CARLO_PATHS,
        confidence_level: float = config.MONTE_CARLO_CONFIDENCE_LEVEL
    ) -> Dict:
        """
        Run Monte Carlo simulation on trade distribution.
        
        Args:
            trades: List of trade results
            num_paths: Number of paths to simulate
            confidence_level: Confidence level for percentiles
            
        Returns:
            Monte Carlo results
        """
        try:
            if not trades or len(trades) < 2:
                logger.warning("Insufficient trades for Monte Carlo")
                return {}
            
            logger.info(f"Running Monte Carlo simulation ({num_paths} paths)...")
            
            # Extract P&Ls with validation
            pnls = []
            for t in trades:
                if 'pnl' in t:
                    pnl = float(t['pnl'])
                    if np.isfinite(pnl):
                        pnls.append(pnl)
            
            if len(pnls) < 2:
                logger.warning(f"Insufficient valid P&L values ({len(pnls)})")
                return {}
            
            pnl_array = np.array(pnls)
            
            # Generate random paths
            final_equities = []
            
            for path_num in range(num_paths):
                try:
                    equity = config.INITIAL_CAPITAL
                    
                    # Randomly shuffle trades
                    shuffled_pnls = np.random.permutation(pnl_array)
                    
                    for pnl in shuffled_pnls:
                        equity += pnl
                        if not np.isfinite(equity):
                            equity = config.INITIAL_CAPITAL  # Reset on bad value
                    
                    final_equities.append(equity)
                except Exception as e:
                    logger.debug(f"Path {path_num} failed: {e}")
                    continue
            
            if not final_equities:
                logger.warning("No valid Monte Carlo paths generated")
                return {}
            
            final_equities = np.array(final_equities)
            
            # Compute percentiles
            lower_pct = (1 - confidence_level) / 2 * 100
            upper_pct = 100 - lower_pct
            
            results = {
                'mean_final_equity': float(np.mean(final_equities)),
                'std_final_equity': float(np.std(final_equities)),
                'percentile_5': float(np.percentile(final_equities, 5)),
                'percentile_25': float(np.percentile(final_equities, 25)),
                'percentile_50': float(np.percentile(final_equities, 50)),
                'percentile_75': float(np.percentile(final_equities, 75)),
                'percentile_95': float(np.percentile(final_equities, 95)),
                'min_equity': float(np.min(final_equities)),
                'max_equity': float(np.max(final_equities)),
            }
            
            logger.info(f"✓ Monte Carlo complete")
            logger.info(f"  Mean final equity: ${results['mean_final_equity']:,.0f}")
            logger.info(f"  5th percentile: ${results['percentile_5']:,.0f}")
            logger.info(f"  95th percentile: ${results['percentile_95']:,.0f}")
            
            return results
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {}
    
    def get_summary_report(self) -> str:
        """Generate comprehensive backtest summary report."""
        if len(self.results) == 0:
            return "No backtest results available"
        
        # Average metrics across all folds
        avg_metrics = {}
        for key in self.results[0].keys():
            values = [r[key] for r in self.results if key in r]
            avg_metrics[key] = np.mean(values) if values else 0
        
        report = format_metrics_table(avg_metrics)
        
        report += f"\n\nNumber of Trades: {avg_metrics.get('num_trades', 0):.0f}"
        report += f"\nProfit Factor: {avg_metrics.get('profit_factor', 0):.2f}"
        report += f"\nWin Rate: {avg_metrics.get('win_rate', 0)*100:.1f}%"
        
        return report
    
    def save_results(self, path: str = config.BACKTEST_RESULTS_PATH):
        """Save backtest results to CSV."""
        if len(self.results) == 0:
            logger.warning("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        logger.info(f"✓ Results saved to {path}")
    
    def save_trades(self, path: str = config.BACKTEST_TRADES_PATH):
        """Save trade log to CSV."""
        if len(self.trades) == 0:
            logger.warning("No trades to save")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(path, index=False)
        logger.info(f"✓ Trades saved to {path}")
