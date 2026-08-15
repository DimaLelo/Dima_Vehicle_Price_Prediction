# 🚗 Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

An end-to-end machine learning pipeline that estimates fair market prices for used vehicles in Kenya. This project addresses the severe information asymmetry in Kenya's informal used-car market by providing a transparent, data-driven valuation tool.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live App:** https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/

## Problem Statement
Kenya's used-vehicle market is dominated by second-hand imports, but pricing remains largely informal and negotiation-driven — there is no shared, data-backed benchmark for what a given vehicle is actually worth. This leads to:

- **Information asymmetry** — sellers and buyers each anchor on different, often unfounded, price expectations
- **Valuation risk** — lenders using vehicles as loan collateral have no reliable way to verify a fair market value
- **Market inefficiency** — prolonged negotiation and mistrust between parties

This project delivers a model that predicts vehicle prices with ~13.78% MAPE, providing a transparent anchor for negotiation and lending decisions.

## Key Features
- **End-to-end pipeline** — from raw data cleaning to a deployed Streamlit app
- **Three-model comparison** — Linear Regression, Random Forest, and XGBoost, with hyperparameter tuning
- **Rare-segment fairness** — explicit evaluation and correction for underrepresented vehicle makes using SMOTE-NC and sample weighting
- **Interpretable AI** — SHAP values confirm predictions are driven by economically sensible features (mileage, age, make-model)
- **Production deployment** — interactive web app with real-time feature engineering and input validation

## Results at a Glance
| Model | RMSE (KES) | MAPE |
|---|---:|---:|
| **XGBoost + SMOTE-NC** ⭐ | **527,308** | **13.78%** |
| XGBoost (tuned) | 519,381 | 13.88% |
| XGBoost (rare-weighted) | 522,768 | 13.96% |
| Random Forest (tuned) | 519,321 | 14.23% |
| Linear Regression (baseline) | 916,218 | 24.70% |

**Model selection:** A paired-samples t-test on 5-fold cross-validation RMSE found no statistically significant difference between tuned Random Forest and tuned XGBoost (t = -1.973, p = 0.120). XGBoost + SMOTE-NC was selected as the final model based on holdout MAPE and rare-segment equity.



⭐ Selected model achieves the lowest overall MAPE and is deployed in the live app.

## Project Structure
```
.
├── Second_Car_Price_Prediction_.ipynb  # Full pipeline: cleaning → modeling
├── app.py                              # Streamlit deployment app
├── Vehicle_Data.xlsx                   # Dataset (46,685 listings)
├── vehicle_price_model.pkl             # Trained XGBoost + SMOTE-NC model
├── vehicle_preprocessor.pkl            # Preprocessing pipeline
├── make_model_encoding.pkl             # Target-encoded Make+Model lookup
├── requirements.txt                    # Python dependencies
├── METHODOLOGY.md                      # Detailed methodology, limitations, future work
├── README.md                           # This file
└── LICENSE                             # MIT License
```

## Quick Start

**1. Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Reproducing the Results

The full pipeline — data cleaning, model training, tuning, SMOTE-NC, significance testing, and segment evaluation — lives in `Second_Car_Price_Prediction_.ipynb`.

**1. Install notebook dependencies**
```bash
pip install xgboost shap imbalanced-learn scikit-optimize openpyxl
```
(The notebook also installs these via `!pip install` in the first cell if using Jupyter/Colab.)

**2. Run all cells in order**
```bash
jupyter notebook Second_Car_Price_Prediction_.ipynb
```
Use "Restart & Run All" (Jupyter) or "Runtime > Run all" (Colab) to ensure all dependencies are created before later cells run. Do not run cells individually or out of order.

**3. Expect hyperparameter tuning to be slow**
The "Hyperparametr Tuning" cell (GridSearchCV for Random Forest + BayesSearchCV for XGBoost) can take several minutes. Do not interrupt it.

**4. Verify your run**

| Step | Expected Output |
|---|---|
| Data cleaning | 31,773 rows remain |
| Linear Regression | RMSE ≈ 916,218 · MAPE ≈ 24.70% |
| XGBoost (tuned) | RMSE ≈ 519,381 · MAPE ≈ 13.88% |
| XGBoost (SMOTE-NC) | RMSE ≈ 527,308 · MAPE ≈ 13.78% |
| Paired t-test | t ≈ -1.973, p ≈ 0.120 |
| Rare-segment MAPE | Common ≈ 13.48% · Rare ≈ 18.58% |

Since `RANDOM_STATE = 42` is fixed throughout, these numbers should reproduce exactly.

## License
This project is licensed under the MIT License — see the `LICENSE` file for details.

## Author
**Daki Lelo Dima**
📧 daksilel@gmail.com

---

For detailed methodology, limitations, and future work, see [METHODOLOGY.md](METHODOLOGY.md).
