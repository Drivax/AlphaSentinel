# Benchmark Run Note

Date: 2026-06-02
Scope: Baseline benchmark snapshot for future model/code comparisons.

## Commands Run

Validation:

```bash
python validate_system.py
```

Bounded backtest:

```bash
python main.py --backtest --start-date 2022-01-01 --end-date 2025-01-01
```

## Key Metrics (from final row of backtest_results.csv)

- total_return: 3.3319890964776285e-08
- cagr: 6.66397832471688e-08
- sharpe_ratio: -239196.5190626892
- sortino_ratio: -15.874507831708042
- calmar_ratio: 0.7054888614518374
- max_drawdown: -4.817886023818852e-08
- volatility: 8.327799075596351e-08
- num_trades: 1
- profit_factor: inf
- win_rate: 1.0
- final_equity: 1000000.033319891

## Artifact Summary

- backtest_results.csv rows: 15
- trades_log.csv rows: 1

## Artifact Hashes (SHA-256)

- backtest_results.csv: f2e43b1c67422415d0259f1dbe145fa4f02a67342330121a29232f9c5aedd8f7
- trades_log.csv: 42f81a40719fae7e12a624954062edc6a87285cc003a5a063e28bba7a0ac4965

## Notes

- This run completed successfully (validation passed 5/5, backtest exit code 0).
- Console output showed Windows cp1252 unicode logging warnings, but artifacts were generated correctly.
