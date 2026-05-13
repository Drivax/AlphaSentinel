# AlphaSentinel Quick Start Guide

Get up and running with the AlphaSentinel trading system in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip or conda
- ~500MB free disk space

## Installation

### Step 1: Clone & Navigate
```bash
cd AlphaSentinel
```

### Step 2: Create Virtual Environment
```bash
# Using venv (Windows)
python -m venv venv
venv\Scripts\activate

# Using venv (Mac/Linux)
python -m venv venv
source venv/bin/activate

# Or using conda
conda create -n alpha python=3.10
conda activate alpha
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation Time:** ~3-5 minutes (depends on internet speed and TensorFlow compilation)

---

## Running AlphaSentinel

### Option 1: Run Full Backtest (Recommended for First Run)

```bash
python main.py --backtest
```

**What it does:**
- Downloads 10 years of historical data for CAC40, DAX, Euro Stoxx 50
- Trains HMM regime detector on historical data
- Trains LSTM neural network for regime classification
- Trains SAC reinforcement learning agent
- Runs walk-forward backtest with realistic transaction costs
- Runs Monte Carlo simulation on trade distribution
- Outputs performance metrics (Sharpe, Sortino, Max DD, Win Rate, etc.)

**Output:**
```
[2025-05-13 10:23:45] Loading data...
[2025-05-13 10:24:12] Training regime detector...
[2025-05-13 10:25:03] Training RL agent...
[2025-05-13 10:26:45] Running walk-forward backtest...
[2025-05-13 10:35:22] ✓ Backtest complete.

╔════════════════════════════════════════════════════════╗
║            PERFORMANCE METRICS SUMMARY                 ║
╠════════════════════════════════════════════════════════╣
║ Annual Return                        14.20%            ║
║ Sharpe Ratio                         1.18              ║
║ Sortino Ratio                        1.87              ║
║ Max Drawdown                        -12.30%            ║
║ Win Rate                             54.60%            ║
║ Profit Factor                        1.84              ║
╚════════════════════════════════════════════════════════╝
```

**Time to Complete:** ~10-15 minutes

---

### Option 2: Paper Trading (Latest Signal)

Generate the latest trading signal based on current market data:

```bash
python main.py --paper-trade
```

**What it does:**
- Downloads latest market data
- Detects current regime (Calm, Volatile, or Crisis)
- Generates LONG/SHORT/NEUTRAL trading signal
- Shows confidence score and anomaly flag

**Output:**
```
╔════════════════════════════════════════════════════════╗
║              TRADING SIGNAL SUMMARY                    ║
╠════════════════════════════════════════════════════════╣
║ Signal:           LONG                         
║ Confidence:        78.5%            
║ Regime:           Calm Trending                
║ Position:          0.500            
║ Regime Multiplier: 1.00x           
║ Anomaly Flag:     NO               
╚════════════════════════════════════════════════════════╝
```

**Time to Complete:** ~2-3 minutes

---

### Option 3: Launch Interactive Dashboard

Launch the Streamlit dashboard for real-time monitoring:

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

**Dashboard Features:**
- 📍 Real-time regime detection with confidence scores
- 🔥 Regime probability heatmap
- 📈 Equity curve and performance metrics
- 🎯 Trading signal recommendations
- 📊 Backtested performance summary

**Time to Complete:** Instant (dashboard loads in ~5 seconds)

---

## Configuration

All parameters are centralized in `config.py`. Common adjustments:

### Risk Management
```python
RISK_PER_TRADE = 0.02              # Risk 2% per signal
MAX_DAILY_LOSS = 0.03              # Halt if -3% in a day
TRANSACTION_COST_BPS = 4.0         # 4 basis points per round-trip
```

### Regime Detection
```python
HMM_STATES = 3                      # 3 regimes (calm, volatile, crisis)
LSTM_LOOKBACK = 60                  # 60-day lookback window
LSTM_EPOCHS = 50                    # Training epochs
```

### Backtesting
```python
WALK_FORWARD_TRAIN_YEARS = 3        # 3-year training window
WALK_FORWARD_TEST_MONTHS = 6        # 6-month test window
MONTE_CARLO_PATHS = 1000            # 1000 simulation paths
```

**Tip:** After changing parameters, delete cached models in `models/` directory to retrain.

---

## Output Files

After running, check these locations:

| File | Location | Content |
|------|----------|---------|
| Backtest Results | `data/backtest_results.csv` | Performance metrics per fold |
| Trade Log | `data/trades_log.csv` | All trades with entry/exit details |
| Logs | `logs/alpha_sentinel.log` | Detailed execution logs |
| Models (HMM) | `models/hmm_model.pkl` | Trained Hidden Markov Model |
| Models (LSTM) | `models/lstm_model.h5` | Trained LSTM network |
| Models (RL) | `models/rl_agent.pkl` | Trained SAC agent |
| Data Cache | `data/cached_prices.pkl` | Cached yfinance data |

---

## Troubleshooting

### Issue: ImportError - tensorflow not found
**Solution:**
```bash
pip install tensorflow --upgrade
```

### Issue: CUDA out of memory (if using GPU)
**Solution:**
```bash
# In config.py, reduce LSTM training:
LSTM_BATCH_SIZE = 16  # Reduced from 32
LSTM_EPOCHS = 25      # Reduced from 50
```

### Issue: yfinance download fails
**Solution:**
```bash
# Clear cache and try again
rm data/cached_prices.pkl
python main.py --backtest
```

### Issue: Dashboard shows old data
**Solution:**
```bash
# Click "Load Data & Detect Regimes" button in sidebar
# Or restart dashboard:
streamlit run app.py
```

---

## Next Steps

1. **Understand the Results:** Read the backtesting report and check equity curve
2. **Analyze Regimes:** Look at regime transitions in dashboard
3. **Tune Parameters:** Adjust `config.py` based on risk tolerance
4. **Paper Trade:** Generate signals for a few weeks, compare with market
5. **Deploy:** Consider live trading on small capital first

---

## Key Concepts

### Market Regimes
- **Calm Trending (State 0):** Low vol, positive drift → trend-following works
- **Volatile (State 1):** High vol, mixed drift → mean-reversion better
- **Crisis (State 2):** Extreme vol, negative drift → defensive, reduce risk

### Why This Works
Traditional strategies assume markets are stationary. They're not. This system:
1. **Detects** market regime changes in real-time
2. **Adapts** trading logic to each regime
3. **Manages** risk dynamically based on market conditions
4. **Monitors** for black swan events (anomalies)

### Expected Performance
- Sharpe Ratio: 0.8–1.5 (good; index is typically 0.3–0.6)
- Sortino Ratio: 1.2–2.0 (excellent downside protection)
- Max Drawdown: -8% to -15% (reasonable for equity strategies)
- Win Rate: 52–58% (slightly above 50%, but with better risk management)

---

## Support & Resources

- **README.md:** Full technical documentation
- **config.py:** Detailed parameter explanations
- **Backtesting Results:** `data/backtest_results.csv`
- **Logs:** `logs/alpha_sentinel.log`

---

## Disclaimer

This project is for **educational and research purposes only**. 

⚠️ **NOT FINANCIAL ADVICE.** Past performance does not guarantee future results. Use at your own risk. Consult with licensed financial advisors before deploying real capital.

---

**Happy Trading! 📈**

For questions or improvements, open an issue on GitHub: https://github.com/Drivax/AlphaSentinel
