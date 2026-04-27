# Retail Sales Forecasting Project

This project focuses on predicting supermarket sales using advanced machine learning techniques, specifically for a Kaggle-style retail dataset.

## Current Best Model
The latest and most accurate model is a **Super Optimized Ensemble** of XGBoost and LightGBM.

### Performance (Validation Set)
- **RMSLE**: 0.39304
- **MAE**: 61.70 (13.21% of mean sales)
- **R²**: 0.9691

## Key Features
- **Advanced Feature Engineering**: Includes holiday flags, oil price indicators, and lagged transactions.
- **Time-Series Lags**: Uses multiple lag periods (16, 17, 18, 21, 28, 35 days) and rolling windows.
- **Target Encoding**: Store-Family level encoding with leakage prevention.
- **Ensemble Blend**: 40/60 weighted blend of XGBoost and LightGBM.

## Project Structure
- `super_optimized_model.py`: The primary training and inference script.
- `features.py`: Shared feature engineering functions.
- `compare_models.py`: Benchmark script comparing Linear Regression, Random Forest, XGB, LGB, and Deep Learning (LSTM/GRU).
- `cross_validate.py`: Walk-forward time-series cross-validation.
- `proposal.md`: Original project proposal.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the optimized model:
```bash
python super_optimized_model.py
```
