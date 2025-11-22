import joblib
import pandas as pd
from .feature_engineering import add_features

def predict(model_path, input_dict):
    model = joblib.load(model_path)
    df = pd.DataFrame([input_dict])
    df = add_features(df)
    return model.predict(df)[0]

# Example usage:
if __name__ == "__main__":
    sample = {"PM2.5": 35, "PM10": 50, "NO2": 20, "SO2": 5, "CO": 0.5, "O3": 10}
    result = predict("../models/rf_aqi_model.pkl", sample)
    print(f"Predicted AQI: {result}")