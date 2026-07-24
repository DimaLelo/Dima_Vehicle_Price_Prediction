# Progress Log

## Completed
- Loaded and inspected raw vehicle listings data
- Handled missing values: dropped rows missing Price/Make/Model/Year;
- Capped extreme prices above 10,000,000 KES; removed remaining outliers using an IQR fence on log-transformed price
- Engineered features: Car_Age, Mileage_Per_Year, Log_Price, Log_Mileage
- Applied out-of-fold K-fold target encoding of Make+Model on Price to avoid data leakage
- Trained a Linear Regression baseline model
-Train Random Forest and XGBoost models
- Evaluate all models using RMSE and MAPE
- Perform rare-segment disaggregated evaluation (common vs. rare vehicle types)
- Run SHAP analysis for model interpretability
## Planned Next
- Deployment
- 
