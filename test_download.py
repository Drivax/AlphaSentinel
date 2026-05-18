#!/usr/bin/env python
"""Quick test of yfinance data download"""
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try downloading one symbol
print("Downloading ^FCHI...")
df = yf.download('^FCHI', start='2020-01-01', end='2025-12-31', progress=False)

print(f"Type of result: {type(df)}")
print(f"Shape: {df.shape if hasattr(df, 'shape') else 'N/A'}")
print(f"Is DataFrame: {isinstance(df, pd.DataFrame)}")
print(f"Is Series: {isinstance(df, pd.Series)}")

if isinstance(df, pd.DataFrame):
    print(f"Columns: {df.columns.tolist()}")
    print(f"First rows:\n{df.head()}")
elif isinstance(df, pd.Series):
    print("Downloaded as Series - converting to DataFrame")
    df_converted = df.to_frame()
    print(f"Converted shape: {df_converted.shape}")
    print(f"Converted columns: {df_converted.columns.tolist()}")
