import pandas as pd

def add_features(df):
    df = df.copy()
    if 'PM2.5' in df.columns:
        df['PM2.5_lag1'] = df['PM2.5'].shift(1)
        df['PM2.5_lag2'] = df['PM2.5'].shift(2)
        df['PM2.5_roll3'] = df['PM2.5'].rolling(window=3, min_periods=1).mean()
    if 'AQI' in df.columns:
        df['AQI_lag1'] = df['AQI'].shift(1)
        df['AQI_roll7'] = df['AQI'].rolling(window=7, min_periods=1).mean()
    df = df.fillna(method='bfill').fillna(0)
    return df