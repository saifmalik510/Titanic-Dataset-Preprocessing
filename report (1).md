# Titanic Dataset — Data Preprocessing Report

## 1. Overview

Source data: the classic Kaggle Titanic `train.csv` (891 passengers, 12
columns). The goal was to turn this raw file into a fully numeric,
leakage-free, model-ready dataset with a reusable Scikit-learn
preprocessing pipeline.

## 2. Initial Exploration

- **Shape:** 891 rows × 12 columns.
- **Target:** `Survived` — 0 = died (549, ~61.6%), 1 = survived (342, ~38.4%).
  Moderately imbalanced, so the train/test split uses stratified sampling.
- **Missing values:**
  - `Age`: 177 missing (~19.9%)
  - `Cabin`: 687 missing (~77.1%)
  - `Embarked`: 2 missing (~0.2%)
- **Duplicates:** none found in the raw data.

## 3. Data Cleaning Decisions

| Column | Decision | Why |
|---|---|---|
| Duplicates | Dropped (0 found) | Standard hygiene step; none were present in this dataset. |
| `Age` | Filled with **median** | Age is right-skewed with outliers (elderly passengers); median is more robust than mean. |
| `Fare` | Filled with **median** | Same reasoning — fare distribution is heavily right-skewed. |
| `Embarked` | Filled with **mode** ("S") | Only 2 missing values; mode is the simplest, safest choice for a categorical field this sparse. |
| `Cabin` | Converted to binary `HasCabin` flag, raw column dropped | 77% missing — too sparse to impute or use as a high-cardinality category, but cabin *presence* is informative (it strongly correlates with ticket class and survival), so a presence indicator preserves that signal without fabricating deck values. |
| `PassengerId` | Dropped | Row identifier, no predictive value, and a very high risk of overfitting if left in. |
| `Ticket` | Dropped | Free-text/alphanumeric identifier with very high cardinality and no consistent structure; not used for feature engineering here. |
| `Name` | Used to extract `Title`, then dropped | The raw name string is an identifier, but the embedded title (Mr/Mrs/Miss/Master/etc.) is a strong, well-known Titanic survival predictor. |

## 4. Feature Engineering

- **`FamilySize` = `SibSp` + `Parch` + 1** — total family members aboard,
  including the passenger. Family size has a non-linear relationship with
  survival (very large families and solo travelers both fared worse than
  small families).
- **`IsAlone`** — binary flag, 1 if `FamilySize == 1`. Captures the "solo
  traveler" effect directly, since it isn't purely linear in `FamilySize`.
- **`Title`** — extracted via regex from `Name` (text between the comma and
  the period). Rare/equivalent titles were consolidated:
  - `Mlle`, `Ms` → `Miss`; `Mme` → `Mrs`
  - `Lady`, `Countess`, `Capt`, `Col`, `Don`, `Dr`, `Major`, `Rev`, `Sir`,
    `Jonkheer`, `Dona` → `Rare`
  - Final categories: `Mr`, `Mrs`, `Miss`, `Master`, `Rare`
- **`HasCabin`** — binary indicator described above, derived while cleaning
  `Cabin`.

## 5. Outlier Handling

Checked `Age` and `Fare` using the IQR method. `Age` had a modest number
of outliers on the high end (elderly passengers) and was left untouched,
since these are legitimate values, not data errors. `Fare` had a long
right tail driven by a small number of very expensive first-class
tickets; rather than deleting these real passengers, `Fare` was **capped
at its 99th percentile** to limit the influence of extreme values on
scaling and downstream models while keeping every row.

## 6. Encoding & Scaling

- **Categorical encoding:** `Sex`, `Embarked`, and `Title` are one-hot
  encoded (`OneHotEncoder(handle_unknown='ignore')`) inside the pipeline.
  One-hot was chosen over label/ordinal encoding for the final pipeline
  because none of these categories have a natural order, and label
  encoding would incorrectly imply one (e.g., that `Embarked=2` is
  "greater than" `Embarked=1`). A label-encoded exploratory copy is also
  included in the notebook/outputs for quick inspection, but the pipeline
  used for modeling uses one-hot encoding.
- **Numeric scaling:** `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, and
  `Pclass` are standardized with `StandardScaler` (zero mean, unit
  variance) — appropriate for distance- and gradient-based models
  (logistic regression, SVM, neural nets) and harmless for tree-based
  models.
- **Binary features** (`IsAlone`, `HasCabin`) are passed through with a
  defensive most-frequent imputer but are not scaled, since they are
  already on a 0/1 scale.

## 7. Data Leakage Prevention

All imputation statistics (medians/modes) and scaling parameters used in
the **modeling pipeline** (Section 6 of the notebook) are computed via
`preprocessor.fit_transform(X_train)` — i.e., fit **only on the training
split** — and then applied to the test split with `preprocessor.transform(X_test)`,
which reuses the training statistics without recomputing them. This
ensures no information about the test set's distribution leaks into
imputation or scaling.

(Note: the median/mode imputation shown in the notebook's Section 2 is
applied to the full dataset for the initial EDA/clean-CSV export and
report exploration only — the actual train/test features used for
modeling go through the leak-safe pipeline in Section 6.)

## 8. Train/Test Split

- 80% train / 20% test
- `random_state=42` for reproducibility
- `stratify=y` to preserve the ~61.6% / 38.4% class balance in both splits

## 9. Validation Checklist

- ✅ No missing values remain in the transformed train/test features
- ✅ All transformed features are numeric
- ✅ No data leakage — pipeline fit only on training data
- ✅ Class distribution preserved across train/test splits (stratified)

## 10. Deliverables

| File | Description |
|---|---|
| `titanic_preprocessing.ipynb` | Full, executed notebook with all steps and outputs |
| `outputs/titanic_clean.csv` | Cleaned, feature-engineered dataset (human-readable, pre-encoding) |
| `outputs/titanic_clean_encoded_scaled.csv` | Same dataset, fully label-encoded and scaled (illustrative) |
| `outputs/X_train.csv`, `outputs/X_test.csv` | Raw (pre-pipeline) train/test feature splits |
| `outputs/y_train.csv`, `outputs/y_test.csv` | Train/test target splits |
| `outputs/X_train_transformed.csv`, `outputs/X_test_transformed.csv` | Final, pipeline-transformed, fully numeric features ready for modeling |
| `outputs/preprocessing_pipeline.joblib` | Fitted Scikit-learn `ColumnTransformer` pipeline (imputers + scaler + one-hot encoder) |
| `outputs/target_distribution.png`, `outputs/outlier_boxplots.png` | EDA charts |
| `report.md` | This report |
