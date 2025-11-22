# Almaty AQI Prediction

This project predicts Air Quality Index (AQI) for Almaty using machine learning models such as Random Forest and XGBoost. It includes data preprocessing, feature engineering (lags, rolling means), model training, evaluation, and a Streamlit-based UI for 3-day AQI forecasting.

## Features

- Data cleaning and AQI calculation from PM2.5
- Lag and rolling window feature engineering
- Model training: Linear Regression, Random Forest, XGBoost, Stacking, and more
- Model evaluation and comparison (MAE, RMSE, R²)
- SHAP explainability for model insights
- Streamlit UI for predicting AQI 3 days ahead using real dataset features

## File Overview

- `air_quality.ipynb` — Main notebook: data processing, feature engineering, model training, evaluation, and model saving.
- `ui.ipynb` — Streamlit interface for AQI prediction using your trained models and dataset.
- `rf_aqi_model.pkl`, `xgb_aqi_model.pkl` — Trained model files (saved with `joblib`).
- `scaler.pkl` — Feature scaler (StandardScaler) used for preprocessing.
- `almaty_air.csv` — Cleaned dataset for predictions (replace with your actual CSV if named differently).

## How to Use

1. **Train and Save Models**
   - Run `air_quality.ipynb` to process data and train models.
   - This will generate `rf_aqi_model.pkl`, `xgb_aqi_model.pkl`, and `scaler.pkl` in your project folder.

2. **Run the Streamlit UI**
   - Make sure `ui.ipynb` or `aqi_predict_ui.py` is present.
   - Install Streamlit if needed:
     ```
     pip install streamlit
     ```
   - Run the UI (if using Python script):
     ```
     streamlit run aqi_predict_ui.py
     ```
   - If using `ui.ipynb`, run it as a regular notebook or convert to script.

3. **Predict AQI**
   - Select a date from your dataset in the UI.
   - The app will predict AQI for the next 3 days using real features and your trained model.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, joblib, streamlit, xgboost, shap

Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib joblib streamlit xgboost shap
```

## Notes

- Ensure all `.pkl` model files and the dataset CSV are in the same directory as your UI script/notebook.
- The UI uses only features from your dataset for predictions (no manual input).
- For best results, regularly retrain your model with updated data.

---

**Author:**  
Damilya Amangeldykyzy
Yasmin Tleukhan