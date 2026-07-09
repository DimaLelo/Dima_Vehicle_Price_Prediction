# Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

This repository contains the data processing pipeline and machine learning models developed for the research project *"Predicting Second-Hand Vehicle Prices in Kenya Using Data-Driven Machine Learning Techniques"* (Daki Lelo Dima, USIU-Africa, 2026).

## Overview

The Kenyan used-car market suffers from information asymmetry and inconsistent pricing. This project builds and compares three machine learning models — **Multiple Linear Regression**, **Random Forest**, and **XGBoost** — to predict used vehicle prices from listings data, with a focus on:

- Kenya-specific feature engineering (vehicle age, mileage per year, county-level effects)
- Fair performance across both common and rare vehicle segments
- Model interpretability using SHAP values

## Data

Data was collected from public vehicle listings on Jiji.co.ke and Cheki Kenya. Raw data is not included in this repository for privacy reasons — see `data/README.md` for details.

| Column | Description |
|---|---|
| Vehicle_Title | Listing title |
| Price | Asking price (KES) |
| Year | Manufacture year |
| Engine_CC | Engine displacement (cc) |
| Transmission | Automatic / Manual (mostly missing, not used in modeling) |
| Mileage | Distance driven (km) |
| Condition | Foreign Used / Locally Used / Brand New |
| Make, Model | Vehicle make and model |
| County | Listing location |

## Repository Structure

```
├── notebooks/
│   └── Second_Car_Price_Prediction.ipynb   # main analysis notebook
├── data/
│   ├── raw/                                # original scraped data (gitignored)
│   ├── processed/                          # cleaned train/test splits
│   └── flagged/                            # rows flagged for manual review
├── plots/                                  # EDA and result figures
├── output/                                 # model outputs, flagged review files
├── requirements.txt
└── README.md
```

## Methodology Summary

1. **Data cleaning** — drop rows missing Price/Make/Model/Year; treat `Engine_CC == 0` as missing; impute via Make+Model median (fallback to Make, then global median); cap prices above 10,000,000 KES; winsorize remaining price outliers using an IQR fence on the log scale; clean and impute `Condition`.
2. **Feature engineering** — `Car_Age`, `Mileage_Per_Year`, `Log_Price`, `Log_Mileage`; out-of-fold K-fold target encoding of `Make_Model` on `Price` to avoid leakage.
3. **Modeling** — Linear Regression (baseline), Random Forest, XGBoost, compared using RMSE and MAPE.
4. **Rare-segment evaluation** — vehicles under a rarity threshold are evaluated separately to check for performance fairness.
5. **Interpretability** — SHAP values used to explain feature contributions to predicted price.

## Setup

```bash
git clone https://github.com/<your-username>/vehicle-price-prediction-kenya.git
cd vehicle-price-prediction-kenya
pip install -r requirements.txt
```

Then open `notebooks/Second_Car_Price_Prediction.ipynb` in Jupyter or Google Colab.

## Status

This project is in progress. Current notebook includes data cleaning, feature engineering, and a Linear Regression baseline. Random Forest, XGBoost, MAPE evaluation, rare-segment analysis, and SHAP interpretability are in development.

## Author

Daki Lelo Dima — United States International University – Africa, School of Science and Technology, BSc Data Science and Analytics.

## License

This project is submitted as part of an academic research proposal. All rights reserved per USIU-Africa copyright requirements.
