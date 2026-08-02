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

.

## Setup

```bash
git clone https://github.com/DimaLelo/Dima_Vehicle_Price_Prediction.git
cd Dima_Vehicle_Price_Prediction
pip install -r requirements.txt
```




