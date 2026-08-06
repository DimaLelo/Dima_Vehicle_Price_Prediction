# Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

This repository contains the data processing pipeline and machine learning models developed for the research project *"Predicting Second-Hand Vehicle Prices in Kenya Using Data-Driven Machine Learning Techniques."*



[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**🔗 Live app:** https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/

---

## Overview

Kenya's used-vehicle market is dominated by second-hand imports, but pricing is largely informal and negotiation-driven — there is no shared, data-backed benchmark for what a given car is actually worth. This project builds an end-to-end pipeline that:

1. Cleans and engineers features from a raw dataset of ~46,700 Kenyan vehicle listings
2. Trains and compares Linear Regression, Random Forest, and XGBoost regressors
3. Tunes the top models (Grid Search / Bayesian optimization) and evaluates them with RMSE and MAPE
4. Explicitly tests whether the model is fair to **rare vehicle makes** (e.g. Porsche, Jaguar, Land Rover) and corrects for imbalance using SMOTE-NC and sample weighting
5. Explains predictions with SHAP
6. Deploys the winning model as an interactive **Streamlit** web app

## Live Demo

Try it here: **https://dimavehiclepriceprediction-mtn2sld8vphnzpwszk9xcw.streamlit.app/**

Enter a car's make, model, condition, transmission, county, year of manufacture, and mileage, and the app returns an estimated price in Kenyan Shillings (KES).

## Project Structure

```
.
├── Second_Car_Price_Prediction.ipynb   # Full data science pipeline (cleaning → modeling → export)
├── app.py                              # Streamlit deployment app                  # Raw dataset (see Dataset section)
├── vehicle_price_model.pkl             # Trained model artifact (exported from notebook)
├── vehicle_preprocessor.pkl            # Fitted preprocessing pipeline  *(see Note on Artifacts below)*
├── make_model_encoding.pkl             # Make+Model target-encoding lookup  *(see Note on Artifacts below)*
├── requirements.txt                    # Python dependencies
└── README.md
```


## Dataset

- **Source:** [Kenya Cars on Sale](https://www.kaggle.com/datasets/rukwarop/kenya-cars-on-sale) (Kaggle), a secondary dataset of Kenyan used-vehicle listings
- **Raw size:** 46,685 listings, 9 columns — `Price`, `Make`, `Model`, `YOM` (Year of Manufacture), `Color`, `Used` (condition), `Transmission`, `Mileage`, `Location`
- **Note:** prices reflect **asking prices**, not confirmed transaction values

## Methodology

### 1. Data Cleaning
- Removed 7,549 duplicate listings
- Canonicalized inconsistent Make/Model spelling (case/whitespace variants mapped to the most frequent spelling) and fixed known two-word brand mis-splits (e.g. "Land" → "Land Rover")
- Filtered structurally corrupted rows (invalid year, condition, transmission, mileage, or price format) — 332 rows dropped (0.85%)
- Dropped rows missing any of `Price`, `Make`, `Model`, `YOM`, or `Mileage` — 6,582 rows dropped (16.96%)
- Capped prices at KES 10,000,000 and trimmed remaining outliers using an IQR filter on log-transformed price

### 2. Feature Engineering
- `Car_Age` = current year − year of manufacture
- `Mileage_Per_Year` = mileage ÷ car age
- `Log_Price`, `Log_Mileage` — log1p transforms to correct right-skew
- `MakeModel_TE` — target-encoded Make+Model combination, computed via **5-fold out-of-fold encoding** on the training set to prevent leakage
- One-hot encoding for `Condition`, `Transmission`, `County`

### 3. Model Training & Tuning
| Model | Tuning method |
|---|---|
| Linear Regression | Baseline, no tuning |
| Random Forest | `GridSearchCV` (3-fold) over `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | `BayesSearchCV` (scikit-optimize, 15 iterations, 3-fold) over `n_estimators`, `max_depth`, `learning_rate`, `subsample` |

### 4. Rare-Segment Fairness
Vehicle makes representing less than 2% of listings (7.69% of the dataset combined — Land Rover, Audi, Lexus, Ford, Daihatsu, Volvo, Peugeot, Jeep, Porsche, Mini, Jaguar, Hyundai, Kia, Chevrolet, Alfa Romeo) were evaluated separately from common makes. Two mitigations were tested:
- **Sample weighting** — 3× weight on rare-make training errors
- **SMOTE-NC** — synthetic oversampling of rare-make records (mixed categorical/continuous features)

### 5. Interpretability
SHAP (`TreeExplainer`) values were computed on the tuned XGBoost model to confirm that predictions are driven by economically sensible features rather than spurious correlations.

## Results

Evaluated on a 20% holdout test set (RMSE and MAPE computed on actual, non-log-transformed KES prices):

| Model | RMSE (KES) | MAPE |
|---|---:|---:|
| **XGBoost (SMOTE-NC)** ⭐ | **526,613** | **13.84%** |
| XGBoost (tuned) | 519,381 | 13.88% |
| XGBoost (rare-weighted) | 522,768 | 13.96% |
| Random Forest (tuned) | 519,321 | 14.23% |
| XGBoost (untuned) | 545,448 | 14.92% |
| Random Forest (untuned) | 526,809 | 14.98% |
| Linear Regression (baseline) | 916,218 | 24.70% |

⭐ **Selected model** — lowest MAPE overall, deployed in the live app.

**Segment-level performance (final model):**

| Segment | n (test set) | RMSE (KES) | MAPE |
|---|---:|---:|---:|
| Common makes | 5,855 | 474,933 | 13.48% |
| Rare makes | 500 | 887,289 | 18.58% |

**Statistical significance (5-fold CV, tuned Random Forest vs. tuned XGBoost):**
t = −1.973, p = 0.120 → no statistically significant difference between the two tuned ensemble models on cross-validated RMSE.

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/DimaLelo/Dima_Vehicle_Price_Prediction.git
cd Dima_Vehicle_Price_Prediction
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate      
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook (optional — to retrain the model)
```bash
jupyter notebook Second_Car_Price_Prediction.ipynb
```

### 5. Launch the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## Tech Stack

- **Language:** Python 3.10+
- **Data processing:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost, scikit-optimize (Bayesian tuning), imbalanced-learn (SMOTE-NC)
- **Interpretability:** SHAP
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Serialization:** joblib

## Limitations

- Prices are **asking prices** from online listings, not confirmed transaction prices
- The underlying dataset does not disclose which specific listing platforms or dealers the records originated from
- Rare-make predictions remain meaningfully less accurate (18.58% MAPE) than common-make predictions (13.48% MAPE), despite oversampling
- The model does not account for vehicle condition beyond "Foreign Used" / "Local Used" (e.g. no accident history, service records, or interior/exterior condition)

## Future Work

- Incorporate transaction-level (not just asking-price) data where available
- Add equipment/trim-level features (e.g. sunroof, leather seats, infotainment)
- Expand geographic coverage beyond well-represented counties
- Add confidence intervals around point predictions

## Author
**<Dima Daki>**
📧 <daksilel@gmail.com>

## Acknowledgments

- Dataset: [Kenya Cars on Sale](https://www.kaggle.com/datasets/rukwarop/kenya-cars-on-sale) by rukwarop, via Kaggle
- Built with [Streamlit](https://streamlit.io/) and [XGBoost](https://xgboost.readthedocs.io/)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
