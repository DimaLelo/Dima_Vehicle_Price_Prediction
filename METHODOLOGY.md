# Methodology, Limitations, and Future Work

## Dataset

| Property | Value |
|---|---|
| Source | Kenya Cars on Sale (Kaggle) |
| Raw records | 46,685 listings, 9 columns |
| Final records | 31,773 (after cleaning and outlier removal) |
| Features | Price, Make, Model, YOM, Condition, Transmission, Mileage, Location, Color |
| Price type | Asking prices (not confirmed transaction values) |

## Methodology

### 1. Data Cleaning

- **Removed 7,549 duplicate listings** — exact row repeats, likely from web scraping re-runs
- **Canonicalized Make/Model spelling** — mapped case/whitespace variants to their most frequent spelling (e.g., "BMW", "Bmw", "bmw" all → "BMW")
- **Fixed known brand mis-splits** — two-word makes incorrectly split at the space during scraping ("Land" → "Land Rover", "Alfa" → "Alfa Romeo")
- **Filtered 332 structurally corrupted rows** (0.85%) — values in wrong columns (e.g., price strings in Model field), detected via validation rules before parsing
- **Dropped 6,582 rows missing critical fields** (Mileage had the highest missingness at 16.96%)
- **Price capping** — removed 394 rows with prices > KES 10,000,000 to reduce outlier influence
- **IQR outlier trimming on log scale** — removed 55 rows outside 1.5×IQR bounds of log-transformed prices

**Final row count:** 31,773

### 2. Feature Engineering

**Numeric features:**
- `Car_Age` = 2026 − year of manufacture
- `Mileage_Per_Year` = mileage ÷ car age
- `Log_Price`, `Log_Mileage` — log1p transforms to correct right-skewed distributions
- `MakeModel_TE` — 5-fold out-of-fold target encoding of Make + Model concatenated, to prevent train/test leakage while encoding a high-cardinality categorical

**Categorical features:**
- `Condition` (Foreign Used, Kenyan Used, Brand New) — one-hot encoded
- `Transmission` (Manual, Automatic, CVT, AMT) — one-hot encoded
- `County` (extracted from Location field) — one-hot encoded

### 3. Model Training & Hyperparameter Tuning

**Linear Regression (baseline)**
- No tuning; provides a lower bound on model performance

**Random Forest (tuned)**
- Grid search with 3-fold cross-validation over:
  - `n_estimators` ∈ [200, 400]
  - `max_depth` ∈ [10, None]
  - `min_samples_leaf` ∈ [1, 5]
- Best params: `n_estimators=400`, `max_depth=None`, `min_samples_leaf=5`
- Test RMSE: 519,321 KES · MAPE: 14.23%

**XGBoost (tuned)**
- Bayesian search (scikit-optimize) with 15 iterations, 3-fold cross-validation over:
  - `n_estimators` ∈ [200, 600]
  - `max_depth` ∈ [3, 10]
  - `learning_rate` ∈ [0.01, 0.3] (log-uniform prior)
  - `subsample` ∈ [0.6, 1.0]
- Best params: `n_estimators=497`, `max_depth=9`, `learning_rate≈0.0101`, `subsample≈0.7018`
- Test RMSE: 519,381 KES · MAPE: 13.88%

### 4. Rare-Segment Fairness Analysis

Vehicle makes representing < 2% of listings (Land Rover, Audi, Lexus, Ford, Daihatsu, Volvo, Peugeot, Jeep, Porsche, Mini, Jaguar, Hyundai, Kia, Chevrolet, Alfa Romeo) were flagged as "rare" and evaluated separately on both common and rare subsets.

**Two mitigations tested:**

**Sample Weighting:**
- Applied 3× weight to XGBoost training errors for rare-make vehicles
- Result: RMSE 522,768 KES · MAPE 13.96%

**SMOTE-NC (final approach):**
- Synthetic oversampling applied to the raw categorical + numeric feature matrix (before one-hot encoding)
- Categorical columns (Condition, Transmission, County, Is_Rare) marked as categorical for SMOTE-NC, so they are resampled by nearest-neighbor majority vote (always valid category values)
- Numeric columns (Car_Age, Log_Mileage, Mileage_Per_Year, MakeModel_TE) are interpolated
- Log_Price included in resampling so synthetic rows get realistic interpolated targets
- Sampling strategy: oversample rare class to 30% of majority class size
- Result: **XGBoost + SMOTE-NC** — RMSE 527,308 KES · MAPE **13.78%**

### 5. Model Comparison & Significance Testing

A paired-samples t-test on 5-fold cross-validation RMSE for tuned Random Forest vs. tuned XGBoost:
- RF CV-RMSE: 0.1974 ± 0.0036
- XGB CV-RMSE: 0.1983 ± 0.0039
- **t-statistic: -1.973**
- **p-value: 0.1198**
- **Result: No statistically significant difference** (p > 0.05)

Despite RF's marginally lower CV RMSE, XGBoost + SMOTE-NC was selected for deployment because:
1. Holdout MAPE (13.78%) is lower than RF (14.23%)
2. XGBoost's compatibility with SMOTE-NC directly addresses rare-segment equity
3. Final model selection balances holdout accuracy, segment-level fairness, and interpretability, not cross-validation performance alone

### 6. Interpretability

SHAP (`TreeExplainer`) values computed on the final XGBoost + SMOTE-NC model confirm that predictions are primarily driven by:
1. **Mileage** (usage intensity)
2. **Car Age** (temporal depreciation)
3. **Make-Model interactions** (brand prestige, reliability, parts availability)

These are economically rational price drivers, suggesting the model captures genuine market dynamics rather than spurious correlations.

---

## Limitations

- **Asking prices vs. transaction prices** — the dataset records listing prices negotiated between seller and buyer, not confirmed transaction values. In Kenya's informal market, realized prices may diverge systematically from listed prices.

- **Unverified data provenance** — the scraping sources, dealer transparency, and listing freshness are undisclosed.

- **Geographic concentration** — data is heavily concentrated in Nairobi, Mombasa, and Kisumu. Rural and semi-urban markets, where road conditions and income levels differ substantially, are underrepresented.

- **Rare-segment error remains elevated** — even after SMOTE-NC, rare vehicles (MAPE 18.58%) are predicted less accurately than common ones (MAPE 13.48%). The 5 percentage-point gap reflects genuine market challenges for niche brands: limited comparable sales, specialist pricing, and hidden condition issues are harder to capture from asking price alone.

- **Limited condition data** — the dataset lacks mechanical reliability assessments, service history, or accident records. Mechanical degradation is widely recognized as a significant hidden cost for imported used vehicles but is invisible to the model.

- **Price ceiling** — vehicles valued above KES 10,000,000 were excluded to reduce outlier influence, rendering the model inapplicable to the luxury segment.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Modeling | scikit-learn, XGBoost, scikit-optimize (Bayesian hyperparameter search), imbalanced-learn (SMOTE-NC) |
| Interpretability | SHAP |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit, Streamlit Community Cloud |
| Serialization | joblib |

---

## Future Work

- **Transaction-level data integration** — partner with dealerships, auction houses, and fintech platforms to access confirmed sale prices and historical transactions, moving beyond asking prices.

- **Equipment and trim features** — scrape or crowdsource data on vehicle condition markers (sunroof, leather seats, infotainment system, air conditioning status) to improve predictions for rare segments.

- **Geographic expansion** — add data from rural and semi-urban markets with explicit modeling of regional variation (road quality, fuel prices, income levels).

- **Macroeconomic indicators** — incorporate time-varying signals such as KES/USD exchange rate, fuel prices, import duty adjustments, and global supply chain indices to capture market shocks.

- **Computer vision for condition** — train a CNN on user-uploaded vehicle photos to estimate cosmetic and mechanical condition, addressing the gap left by missing inspection data.

- **Deep learning exploration** — experiment with entity embeddings for Make/Model, stacking ensembles, and neural architectures on larger datasets.

- **Prediction confidence intervals** — extend the app to output uncertainty bands alongside point predictions, supporting risk-aware lending and negotiation.

- **A/B testing in production** — once deployed, gather actual transaction data through user feedback to continuously audit and recalibrate the model's predictions in the real market.
