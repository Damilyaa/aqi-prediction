import joblib
from sklearn.metrics import mean_squared_error
from .data_preprocessing import load_data, preprocess_data
from .feature_engineering import add_features

def evaluate_model(model_path, data_path):
    df = load_data(data_path)
    df = add_features(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"MSE: {mse}")

if __name__ == "__main__":
    evaluate_model("../models/rf_aqi_model.pkl", "../data/almaty-air.csv")