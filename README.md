# Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

This repository contains the data processing pipeline and machine learning models developed for the research project *"Predicting Second-Hand Vehicle Prices in Kenya Using Data-Driven Machine Learning Techniques."*

## Table of Contents
- [Project Overview](#project-overview)
- [Author](#author)
- [Problem Statement](#problem-statement)
- [Research Questions](#research-questions)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Setup](#setup)
- [Current Status / Progress Summary](#current-status--progress-summary)
- [Planned / Future Work](#planned--future-work)

## Project Overview

This project builds a machine learning model to predict the asking price of second-hand vehicles in Kenya, using listings data scraped from public online vehicle marketplaces. It compares three modeling approaches — Multiple Linear Regression, Random Forest, and XGBoost — and pays particular attention to Kenya-specific features (vehicle age, mileage per year, county) and to whether the model performs fairly on both common and rare vehicle types.

## Author

Daki Lelo Dima

## Problem Statement

Kenya's used-car market has no standardized way to value a vehicle. Prices are set through informal negotiation rather than data, because pricing models built for other countries don't account for how vehicles depreciate on Kenyan roads or how the market is dominated by imports from Japan, the UK, and the UAE. This creates real costs: buyers and sellers can't tell if a price is fair, and banks struggle to value vehicles used as loan collateral.

## Research Questions

1. Which vehicle features (mileage, age, county, etc.) have the biggest influence on used car prices in Kenya?
2. Which machine learning approach — Linear Regression, Random Forest, or XGBoost — predicts Kenyan used car prices most accurately?
3. Does the model perform equally well for rare vehicle makes as it does for common ones?

## Objectives

1. Identify the vehicle features that most affect price in Kenya, using correlation analysis, Random Forest feature importance, and SHAP values.
2. Train and compare Linear Regression, Random Forest, and XGBoost, selecting the best model using RMSE, MAPE, and R².
3. Evaluate whether the model is equally accurate for common and rare vehicle segments (makes with under 2% market share).

## Dataset

Data was collected from public used-vehicle listings on Kenyan online marketplaces.

| Column | Description |
|---|---|
| Price | Asking price, scraped as text (e.g. `"KSh 2,050,000"`) and parsed to a numeric KES value |
| YOM | Year of manufacture |
| Mileage | Distance driven (km) |
| Used | Condition — Foreign Used / Kenyan Used / Brand New |
| Transmission | Automatic / Manual / CVT / AMT |
| Make, Model | Vehicle make and model |
| Color | Vehicle color |
| Location | Listing location, parsed down to County |

Raw data is not included in this repository; see `data/README.md` for details on scope and access.

## Repository Structure

```
├── Vehicle_Price_Prediction.ipynb   
├── plots/                           
├── output/                          
├── docs/
│   └── progress.md                  
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

1. **Data cleaning** — remove exact and near-duplicate listings (matched on Make, Model, YOM, Mileage, and Price after normalizing text case); parse `Price`, `Mileage`, and `YOM` to numeric, coercing malformed values to missing; drop rows missing any critical field (Make, Model, YOM, Price, Mileage); restrict to plausible ranges (YOM 1960–2026, Price > 0, Mileage ≥ 0); normalize categorical text fields (Make, Model, Color, Used, Transmission, Location) and fill remaining categorical gaps with `"Unknown"`.
2. **Feature engineering** — `County` parsed from `Location`; `CarAge` (2026 − YOM); `MileagePerYear`; log-transformed `LogPrice` and `LogMileage` to reduce skew; vehicle makes with under 2% market share flagged as a `RareSegment` for later fairness evaluation.
3. **Modeling** — categorical features (Make, Model, Color, Used, Transmission, County) one-hot encoded and numeric features median-imputed inside a scikit-learn pipeline. Linear Regression serves as the baseline; Random Forest is tuned via grid search; XGBoost is tuned via randomized search. All three are compared with 5-fold cross-validation on RMSE.
4. **Evaluation** — final models scored on a held-out test set (stratified by `RareSegment`) using RMSE, MAPE, and R², after transforming predictions back from log scale.
5. **Rare-segment evaluation** — the best-performing model's MAPE is broken out separately for `Rare` vs. `Common` vehicle makes to check for performance disparities.
6. **Interpretability** — correlation analysis, Random Forest feature importances, and SHAP values (via `TreeExplainer` on a sample of the training set) explain each feature's contribution to predicted price.

## Setup

```bash
git clone https://github.com/DimaLelo/Dima_Vehicle_Price_Prediction.git
cd Dima_Vehicle_Price_Prediction
pip install -r requirements.txt
```

Then open `Vehicle_Price_Prediction.ipynb` in Jupyter or Google Colab. Place `Vehicle_Data.xlsx` in the project root before running.

## Current Status / Progress Summary

- [x] Data cleaning (duplicate removal, missing values, invalid-row filtering)
- [x] Feature engineering (CarAge, MileagePerYear, log transforms, rare-segment flagging)
- [x] Linear Regression baseline trained
- [x] Random Forest model (untuned and tuned via grid search)
- [x] XGBoost model (untuned and tuned via randomized search)
- [x] Model comparison via RMSE, MAPE, and R²
- [x] Rare-segment disaggregated MAPE evaluation
- [x] Random Forest feature importance analysis
- [x] SHAP interpretability analysis
- [x] Actual vs. predicted price visualization

Next step: finalize the written report summarizing results and answering the three research questions.

## Planned / Future Work

These techniques are part of the broader research design but are not yet implemented in the current notebook:

- Structural-corruption detection for scraped rows (values shifted into the wrong column) prior to parsing
- Price capping and IQR-based winsorizing on the log-price scale
- Out-of-fold K-fold target encoding of Make + Model (in place of one-hot encoding)
- Bayesian hyperparameter optimization for XGBoost (`skopt`)
- Rebalancing techniques for rare segments (sample weighting, SMOTE-NC with KNN-imputed targets)
- Paired t-test on cross-validated RMSE to test whether the Random Forest vs. XGBoost difference is statistically significant
