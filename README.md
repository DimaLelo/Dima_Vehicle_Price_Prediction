# 🚗 Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

An end-to-end machine learning pipeline that estimates fair market prices for used vehicles in Kenya. This project addresses the severe information asymmetry in Kenya's informal used-car market by providing a transparent, data-driven valuation tool.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live App:** https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/

##  Problem Statement
Kenya's used-vehicle market is dominated by second-hand imports, but pricing remains largely informal and negotiation-driven — there is no shared, data-backed benchmark for what a given vehicle is actually worth. This leads to:

- **Information asymmetry** — sellers and buyers each anchor on different, often unfounded, price expectations
- **Valuation risk** — lenders using vehicles as loan collateral have no reliable way to verify a fair market value
- **Market inefficiency** — prolonged negotiation and mistrust between parties

This project delivers a model that predicts vehicle prices with ~13.84% MAPE, providing a transparent anchor for negotiation and lending decisions.

##  Key Features
- **End-to-end pipeline** — from raw data cleaning to a deployed Streamlit app
- **Three-model comparison** — Linear Regression, Random Forest, and XGBoost, with hyperparameter tuning
- **Rare-segment fairness** — explicit evaluation and correction for underrepresented vehicle makes (e.g., Porsche, Land Rover, Jaguar) using SMOTE-NC and sample weighting
- **Interpretable AI** — SHAP values confirm predictions are driven by economically sensible features (mileage, age, make-model)
- **Production deployment** — interactive web app with real-time feature engineering and input validation

##  Results at a Glance
| Model | RMSE (KES) | MAPE |
|---|---:|---:|
| **XGBoost + SMOTE-NC** ⭐ | **526,613** | **13.84%** |
| XGBoost (tuned) | 519,381 | 13.88% |
| Random Forest (tuned) | 519,321 | 14.23% |
| Linear Regression (baseline) | 916,218 | 24.70% |

**Segment performance (final model):**

| Segment | Test Size | RMSE (KES) | MAPE |
|---|---:|---:|---:|
| Common makes | 5,855 | 474,933 | 13.48% |
| Rare makes | 500 | 887,289 | 18.58% |

⭐ Selected model achieves the lowest overall MAPE and is deployed in the live app. SMOTE-NC narrows the rare-segment error gap but does not fully eliminate it.

##  Project Structure
```
.
├── Second_Car_Price_Prediction.ipynb   # Full pipeline: cleaning → modeling → export
├── app.py                              # Streamlit deployment app
├── vehicle_price_model.pkl             # Trained XGBoost model (exported from notebook)
├── vehicle_preprocessor.pkl            # Fitted preprocessing pipeline
├── make_model_encoding.pkl             # Target-encoded Make+Model lookup table
├── requirements.txt                    # Python dependencies
└── README.md
```
**Note on artifacts:** The `.pkl` files are exported from the notebook after training. Do not retrain unless you intend to modify the modeling approach.

##  Dataset
| Property | Value |
|---|---|
| Source | Kenya Cars on Sale (Kaggle) |
| Raw records | 46,685 listings, 9 columns |
| Final records | 32,222 (after removing duplicates, corrupted rows, and rows missing critical fields) |
| Features | Price, Make, Model, YOM, Condition, Transmission, Mileage, Location, Color |
| Price type | Asking prices (not confirmed transaction values) |

##  Methodology

**1. Data Cleaning**
- Removed 7,549 duplicate listings
- Canonicalized inconsistent Make/Model spelling (case/whitespace variants, known brand mis-splits like "Land" → "Land Rover")
- Filtered 332 structurally corrupted rows (invalid year, condition, transmission, mileage, or price format)
- Dropped 6,582 rows missing critical fields (Price, Make, Model, YOM, Mileage)
- Capped prices at KES 10,000,000 and applied IQR-based outlier trimming on log-transformed prices

**2. Feature Engineering**
- `Car_Age` = current year − year of manufacture
- `Mileage_Per_Year` = mileage ÷ car age
- `Log_Price`, `Log_Mileage` — log1p transforms to correct right-skew
- `MakeModel_TE` — 5-fold out-of-fold target encoding of Make+Model to prevent leakage
- One-hot encoding for `Condition`, `Transmission`, `County`

**3. Model Training & Tuning**

| Model | Tuning Strategy |
|---|---|
| Linear Regression | Baseline (no tuning) |
| Random Forest | `GridSearchCV` (3-fold) over `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | `BayesSearchCV` (scikit-optimize, 15 iterations, 3-fold) over `n_estimators`, `max_depth`, `learning_rate`, `subsample` |

**4. Rare-Segment Fairness**
Vehicle makes representing under 2% of listings (Land Rover, Audi, Lexus, Ford, Daihatsu, Volvo, Peugeot, Jeep, Porsche, Mini, Jaguar, Hyundai, Kia, Chevrolet, Alfa Romeo) were evaluated separately. Two mitigations were tested:
- **Sample weighting** — 3× weight on rare-make training errors
- **SMOTE-NC** — synthetic oversampling for mixed categorical/continuous features

**5. Interpretability**
SHAP (`TreeExplainer`) values computed on the tuned XGBoost model confirm that predictions are driven by mileage, vehicle age, and make-model — economically rational features rather than spurious correlations.

##  How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/DimaLelo/Dima_Vehicle_Price_Prediction.git
cd Dima_Vehicle_Price_Prediction
```

**2. Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. (Optional) Retrain the model**
```bash
jupyter notebook Second_Car_Price_Prediction.ipynb
```

**5. Launch the app**
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

##  Tech Stack
| Category | Technologies |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Modeling | scikit-learn, XGBoost, scikit-optimize, imbalanced-learn (SMOTE-NC) |
| Interpretability | SHAP |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit, Streamlit Community Cloud |
| Serialization | joblib |

## Limitations
- **Asking prices only** — the dataset reflects listing prices, not confirmed transaction values
- **Unverified provenance** — the underlying platforms and dealers sourcing the listings are undisclosed
- **Rare-segment error** — predictions for uncommon makes remain less accurate (18.58% MAPE vs. 13.48% for common makes)
- **Limited condition data** — no accident history, service records, or mechanical inspection data

##  Future Work
- Integrate transaction-level data from dealerships and marketplaces
- Add equipment/trim-level features (sunroof, leather seats, infotainment)
- Expand geographic coverage to rural and semi-urban counties
- Incorporate macroeconomic indicators (KES/USD exchange rate, fuel prices, import duties)
- Explore deep learning architectures (entity embeddings, stacking ensembles)
- Add prediction confidence intervals to the app output

##  License
This project is licensed under the MIT License — see the `LICENSE` file for details.

##  Author
**Daki Lelo Dima**
📧 daksilel@gmail.com
🔗 [GitHub](https://github.com/DimaLelo)

