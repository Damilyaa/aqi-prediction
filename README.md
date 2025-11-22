# AQI Prediction Project

This project predicts the Air Quality Index (AQI) for Almaty using advanced machine learning techniques. The pipeline covers data cleaning, feature engineering, model training, evaluation, and prediction. The project is modular, with reusable code for each stage, and supports both script-based and notebook-based workflows.

---

## Project Structure

```
aqi-prediction/
│
├── config/                # Configuration files
│   ├── settings.example.yaml
│   └── settings.yaml
│
├── data/                  # Data directories
│   ├── raw/               # Raw input data
│   ├── processed/         # Cleaned and processed data
│   └── external/          # External datasets
│
├── models/                # Trained model files (e.g., rf_aqi_model.pkl, xgb_aqi_model.pkl, scaler.pkl)
│
├── src/                   # Source code
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature engineering (lags, rolling means, etc.)
│   ├── train.py                  # Model training scripts
│   ├── evaluate.py               # Model evaluation scripts
│   ├── predict.py                # Model inference scripts
│   ├── utils.py                  # Utility functions
│   └── __init__.py
│
├── notebooks/             # Jupyter notebooks for exploration and main modeling
│   └── air_quality.ipynb  # Main notebook with final model code and experiments
│
├── tests/                 # Test suites
│
├── docs/                  # Documentation
│
├── requirements.txt       # Python dependencies
└── README.md              # Project overview (this file)
```

---

## Key Features

- **Data Cleaning & Preprocessing:** Handles missing values, outliers, and prepares data for modeling.
- **Feature Engineering:** Adds lag features, rolling statistics, and other domain-specific features to improve model performance.
- **Model Training:** Supports Random Forest, XGBoost, and other regression models.
- **Model Evaluation:** Provides metrics such as MAE, RMSE, and R² for model comparison.
- **Prediction Pipeline:** Script and notebook-based prediction for new data.
- **Modular Codebase:** All core logic is in `src/` for easy reuse and testing.
- **Jupyter Notebook:** All final model code, experiments, and results are documented in `notebooks/air_quality.ipynb`.

---

## How to Use

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place raw data in `data/raw/`.
   - Use scripts in `src/` or the notebook to process and clean data.

3. **Feature Engineering:**
   - Run `src/feature_engineering.py` or use the functions in your pipeline.

4. **Train models:**
   - Use `src/train.py` for script-based training.
   - Or, run all steps interactively in `notebooks/air_quality.ipynb`.

5. **Evaluate models:**
   - Use `src/evaluate.py` or the notebook for evaluation.

6. **Make predictions:**
   - Use `src/predict.py` for script-based inference.

---

## Main Models and Experiments

**The final, main model code and all experiments are documented in the Jupyter notebook:**

```
notebooks/air_quality.ipynb
```

This notebook contains:
- Full data exploration and visualization
- Feature engineering steps
- Model training and hyperparameter tuning
- Evaluation and comparison of different models
- Final model selection and saving

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, joblib, matplotlib

---

## Author

Damilya Amangeldykyzy
Yasmin Tleukhan

---
