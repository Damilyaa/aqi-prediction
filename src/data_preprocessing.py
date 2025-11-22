import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    X = df.drop('AQI', axis=1)
    y = df['AQI']
    return train_test_split(X, y, test_size=0.2, random_state=42)