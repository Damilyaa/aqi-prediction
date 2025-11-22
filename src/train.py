import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .data_preprocessing import load_data, preprocess_data
from .feature_engineering import add_features

def train_models(data_path, model_dir):
    df = load_data(data_path)
    df = add_features(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{model_dir}/rf_aqi_model.pkl")

    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, f"{model_dir}/xgb_aqi_model.pkl")

    print("Models trained and saved.")

if __name__ == "__main__":
    train_models("../data/almaty-air.csv", "../models")