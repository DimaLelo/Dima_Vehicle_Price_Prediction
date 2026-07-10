# Predicting Second-Hand Vehicle Prices in Kenya Using Machine Learning

<<<<<<< HEAD
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
=======
This repository contains the data processing pipeline and machine learning models developed for the research project *"Predicting Second-Hand Vehicle Prices in Kenya Using Data-Driven Machine Learning Techniques"* 
>>>>>>> 471b15403926c2ab9bfbb777d960a11b283773bb

## Project Overview

This project builds a machine learning model to predict the asking price of second-hand vehicles in Kenya, using listings data scraped from public online vehicle marketplaces. It compares three modeling approaches — Multiple Linear Regression, Random Forest, and XGBoost — and pays particular attention to Kenya-specific features (vehicle age, mileage per year, county) and to whether the model performs fairly on both common and rare vehicle types.

## Author

Daki Lelo Dima 

## Problem Statement

Kenya's used-car market has no standardized way to value a vehicle. Prices are set through informal negotiation rather than data, because pricing models built for other countries don't account for how vehicles depreciate on Kenyan roads or how the market is dominated by imports from Japan, the UK, and the UAE. This creates real costs: buyers and sellers can't tell if a price is fair, and banks struggle to value vehicles used as loan collateral.

## Research Questions

1. Which vehicle features (mileage, age, county, etc.) have the biggest influence on used car prices in Kenya?
2. Which machine learning approach — Linear Regression, Random Forest, or XGBoost — predicts Kenyan used car prices most accurately?
3. Does the model perform equally well for rare vehicle makes/models as it does for common ones?

## Objectives

1. Identify the vehicle features that most affect price in Kenya, using correlation analysis and Random Forest feature importance.
2. Train and compare Linear Regression, Random Forest, and XGBoost, selecting the best model using RMSE and MAPE.
3. Evaluate whether the model is equally accurate for common and rare vehicle segments, applying rebalancing techniques if needed.

## Dataset

Data was collected from public used-vehicle listings on Kenyan online marketplaces.

| Column | Description |
|---|---|
| Vehicle_Title | Listing title |
| Price | Asking price (KES) |
| Year | Manufacture year |
| Engine_CC | Engine displacement (cc) |
| Mileage | Distance driven (km) |
| Condition | Foreign Used / Locally Used / Brand New |
| Make, Model | Vehicle make and model |
| County | Listing location |

Raw data is not included in this repository; see `data/README.md` for details on scope and access.

## Repository Structure

```
├── Second_Car_Price_prediction.ipynb   # main analysis notebook
├── plots/                              # EDA and result figures
├── output/                             # flagged rows, model outputs
├── docs/
│   └── progress.md                     # progress log
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

1. **Data cleaning** — drop rows missing Price/Make/Model/Year; treat `Engine_CC == 0` as missing; impute via Make+Model median (fallback to Make, then global median); cap prices above 10,000,000 KES; winsorize remaining price outliers using an IQR fence on the log scale; clean and impute `Condition`.
2. **Feature engineering** — `Car_Age`, `Mileage_Per_Year`, `Log_Price`, `Log_Mileage`; out-of-fold K-fold target encoding of Make+Model on `Price` to avoid leakage.
3. **Modeling** — Linear Regression (baseline), Random Forest, XGBoost, compared using RMSE and MAPE.
4. **Rare-segment evaluation** — vehicles below a rarity threshold are evaluated separately to check for performance fairness.
5. **Interpretability** — SHAP values explain each feature's contribution to predicted price.

## Setup

```bash
git clone https://github.com/DimaLelo/Dima_Vehicle_Price_Prediction.git
cd Dima_Vehicle_Price_Prediction
pip install -r requirements.txt
```

Then open `Second_Car_Price_prediction.ipynb` in Jupyter or Google Colab.

## Current Status / Progress Summary

- [x] Data cleaning (missing values, outlier handling, price capping)
- [x] Feature engineering (Car_Age, Mileage_Per_Year, log transforms, target encoding)
- [x] Linear Regression baseline trained
- [ ] Random Forest model
- [ ] XGBoost model
- [ ] MAPE evaluation alongside RMSE
- [ ] Rare-segment disaggregated evaluation
- [ ] SHAP interpretability analysis

Next step: add Random Forest and XGBoost models and compare against the Linear Regression baseline using RMSE and MAPE.
