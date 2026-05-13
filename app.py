"""
AlphaSentinel Streamlit Dashboard
Real-time monitoring of regime detection and trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
from pathlib import Path

import config
from src.data_loader import load_price_data, extract_regime_features
from src.regime_detector import RegimeDetector
from src.rl_agent import RLAgent
from src.signal_generator import SignalGenerator, PortfolioManager
from src.utils import logger

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="AlphaSentinel - Regime Detection Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1f1f1f;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    .signal-long {
        background-color: #0d5a3d;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ff00;
        color: #00ff00;
        font-weight: bold;
        font-size: 18px;
    }
    .signal-short {
        background-color: #5a0d0d;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff0000;
        color: #ff0000;
        font-weight: bold;
        font-size: 18px;
    }
    .signal-neutral {
        background-color: #3a3a3a;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffaa00;
        color: #ffaa00;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ AlphaSentinel Control Panel")

# Refresh interval
refresh_interval = st.sidebar.slider(
    "Dashboard Refresh Interval (seconds)",
    min_value=60,
    max_value=1800,
    value=300,
    step=60
)

# Symbol selection
symbols_display = st.sidebar.multiselect(
    "Select Markets to Monitor",
    options=list(config.SYMBOL_NAMES.values()),
    default=list(config.SYMBOL_NAMES.values())
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

# Risk parameters
st.sidebar.subheader("Risk Parameters")
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0) / 100
max_daily_loss = st.sidebar.slider("Max Daily Loss (%)", min_value=1.0, max_value=10.0, value=3.0) / 100

# Load data button
load_data = st.sidebar.button("🔄 Load Data & Detect Regimes", use_container_width=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 AlphaSentinel: Market Regime Detection & Trading Dashboard")
st.markdown("Real-time regime detection using HMM + LSTM + Isolation Forest")

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

@st.cache_resource
def load_and_process_data(_symbols, _start_date, _end_date):
    """Load data and train regime detector."""
    try:
        # Load prices
        symbol_list = [k for k, v in config.SYMBOL_NAMES.items() if v in _symbols]
        data = load_price_data(symbol_list, str(_start_date), str(_end_date))
        
        if len(data) == 0:
            st.error("Failed to load price data")
            return None
        
        # Combine multi-asset data
        from src.data_loader import align_multiasset_data, compute_equal_weight_index
        aligned_prices = align_multiasset_data(data)
        portfolio_value = compute_equal_weight_index(aligned_prices)
        
        # Extract features
        features = extract_regime_features(
            pd.DataFrame({'Close': portfolio_value}),
            lookback=config.VOLATILITY_WINDOW
        )
        
        # Train or load regime detector
        detector = RegimeDetector()
        if config.USE_PRETRAINED_HMM or config.USE_PRETRAINED_LSTM:
            try:
                detector.load()
            except:
                logger.warning("Pretrained models not found. Training fresh...")
                detector.fit(features)
        else:
            detector.fit(features)
        
        return {
            'data': data,
            'prices': portfolio_value,
            'features': features,
            'detector': detector
        }
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data if button clicked or on first run
if load_data or 'data_cache' not in st.session_state:
    st.info("Loading market data and training regime detector...")
    data_cache = load_and_process_data(symbols_display, start_date, end_date)
    if data_cache:
        st.session_state.data_cache = data_cache
        st.success("✓ Data loaded successfully")

# ============================================================================
# DISPLAY REGIME DETECTION RESULTS
# ============================================================================

if 'data_cache' in st.session_state:
    cache = st.session_state.data_cache
    detector = cache['detector']
    features = cache['features']
    prices = cache['prices']
    
    # Predict regimes for entire series
    regime_predictions = detector.predict_timeseries(features)
    
    # Get latest regime
    latest_regime = regime_predictions.iloc[-1]
    
    # ========== TOP METRICS ROW ==========
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Regime",
            latest_regime['regime_name'],
            f"Confidence: {latest_regime['confidence']*100:.1f}%"
        )
    
    with col2:
        signal_proba = latest_regime['probabilities']
        st.metric(
            "Regime Probability",
            f"{max(signal_proba)*100:.1f}%",
            "Ensemble confidence"
        )
    
    with col3:
        is_anomaly = "⚠️ YES" if latest_regime['is_anomaly'] else "✓ NO"
        st.metric("Anomaly Detected", is_anomaly)
    
    with col4:
        current_price = prices.iloc[-1]
        prev_price = prices.iloc[-2] if len(prices) > 1 else current_price
        change_pct = (current_price - prev_price) / prev_price * 100
        st.metric(
            "Portfolio Value",
            f"${current_price:.2f}",
            f"{change_pct:+.2f}%"
        )
    
    # ========== REGIME MONITORING ==========
    st.subheader("📍 Regime Detection Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Regime time series
        fig_regime = go.Figure()
        
        fig_regime.add_trace(go.Scatter(
            x=regime_predictions.index,
            y=regime_predictions['regime'],
            mode='lines',
            name='Predicted Regime',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy'
        ))
        
        fig_regime.update_layout(
            title="Regime Transitions (Last 252 Days)",
            xaxis_title="Date",
            yaxis_title="Regime (0=Calm, 1=Volatile, 2=Crisis)",
            hovermode='x unified',
            height=350,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_regime, use_container_width=True)
    
    with col2:
        # Regime distribution
        regime_counts = regime_predictions['regime'].value_counts().sort_index()
        regime_names = ["Calm", "Volatile", "Crisis"]
        
        fig_dist = go.Figure(data=[
            go.Bar(
                x=[regime_names[i] for i in regime_counts.index],
                y=regime_counts.values,
                marker=dict(color=['#2eb8b8', '#ff9900', '#ff3333'])
            )
        ])
        
        fig_dist.update_layout(
            title="Regime Distribution",
            xaxis_title="Regime",
            yaxis_title="Days",
            height=350,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # ========== PROBABILITY HEATMAP ==========
    st.subheader("🔥 Regime Probability Evolution (Last 100 Days)")
    
    recent_regimes = regime_predictions.tail(100).copy()
    proba_matrix = np.array([p for p in recent_regimes['probabilities']])
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=proba_matrix.T,
        x=recent_regimes.index,
        y=["Calm", "Volatile", "Crisis"],
        colorscale='Viridis',
        colorbar=dict(title="Probability")
    ))
    
    fig_heatmap.update_layout(
        title="Regime Probability Heatmap",
        xaxis_title="Date",
        yaxis_title="Regime",
        height=250,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ========== PORTFOLIO PERFORMANCE ==========
    st.subheader("📈 Portfolio Performance")
    
    returns = np.log(prices / prices.shift(1)).dropna()
    cumulative_returns = (1 + returns).cumprod()
    
    fig_equity = go.Figure()
    
    fig_equity.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=(cumulative_returns - 1) * 100,
        mode='lines',
        name='Return',
        line=dict(color='#00ff00', width=2),
        fill='tozeroy'
    ))
    
    fig_equity.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        height=350,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # ========== KEY METRICS ==========
    st.subheader("📊 Performance Metrics")
    
    from src.utils import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_max_drawdown,
        calculate_annual_return,
        calculate_win_rate
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    annual_return = calculate_annual_return(returns)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, _ = calculate_max_drawdown(cumulative_returns)
    
    with col1:
        st.metric("Annual Return", f"{annual_return*100:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col3:
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
    with col5:
        st.metric("Volatility", f"{returns.std()*np.sqrt(252)*100:.2f}%")
    
    # ========== SIGNAL GENERATION ==========
    st.subheader("🎯 Trading Signal (Simulated)")
    
    # Simple signal based on regime
    regime = latest_regime['regime']
    confidence = latest_regime['confidence']
    
    if confidence > 0.65:
        if regime == 0:  # Calm - go long
            signal = "LONG"
            signal_color = "#00ff00"
        elif regime == 1:  # Volatile - neutral
            signal = "NEUTRAL"
            signal_color = "#ffaa00"
        else:  # Crisis - go short or flat
            signal = "NEUTRAL"
            signal_color = "#ffaa00"
    else:
        signal = "NEUTRAL"
        signal_color = "#ffaa00"
    
    signal_html = f'<div class="signal-{signal.lower()}">{signal} | Confidence: {confidence*100:.1f}%</div>'
    st.markdown(signal_html, unsafe_allow_html=True)
    
    # ========== DATA TABLE ==========
    st.subheader("📋 Recent Regime Data")
    
    display_df = regime_predictions.tail(10)[['regime_name', 'confidence', 'is_anomaly']].copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
    display_df['is_anomaly'] = display_df['is_anomaly'].apply(lambda x: "⚠️" if x else "✓")
    
    st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 Click 'Load Data & Detect Regimes' in the sidebar to get started")

# ========== FOOTER ==========
st.divider()
st.markdown("""
**AlphaSentinel** - Professional Regime Detection & Trading System  
📊 Using HMM + LSTM + Isolation Forest for adaptive market analysis  
⚠️ For research and educational purposes only. Not financial advice.
""")
