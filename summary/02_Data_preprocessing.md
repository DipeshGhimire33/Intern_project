# Data Preprocessing Summary

## 1. Preprocessing Objective

The preprocessing stage transformed the original clinical time-series dataset into a compact feature set suitable for machine-learning model development.

The main objectives were to:

* Prevent patient-level data leakage
* Preserve the temporal structure of the observations
* Handle missing clinical measurements
* Reduce the dimensionality of the raw clinical variables
* Incorporate clinically meaningful information
* Preserve useful temporal and demographic information
* Produce consistent training, validation, and test datasets

---

## 2. Patient-Level Train/Validation/Test Split

Because each patient contributes multiple observations, splitting individual rows randomly could result in observations from the same patient appearing in multiple datasets.

To prevent this, patients were divided first and observations were then assigned according to their patient identifier.

The final split was approximately:

* **70% training patients**
* **10% validation patients**
* **20% test patients**

The resulting datasets contained:

| Dataset    | Observations | Non-Sepsis | Sepsis |
| ---------- | -----------: | ---------: | -----: |
| Train      |    1,086,991 |  1,067,723 | 19,268 |
| Validation |      155,226 |    152,450 |  2,776 |
| Test       |      309,993 |    304,121 |  5,872 |

The test patients were completely separated from the training and validation patients.

This patient-level split was maintained throughout model development.

---

## 3. Data Ordering

Within each dataset, observations were sorted by:

1. `Patient_ID`
2. `Hour`

The indexes were then reset.

This ordering was required before applying patient-wise forward filling so that previous observations belonged to the correct patient and occurred earlier in time.

---

## 4. Patient-Wise Forward Filling

Clinical measurements in ICU time-series data are often recorded intermittently.

To preserve information from previous measurements, forward filling was performed within each patient.

The following clinical variables were forward-filled:

* Vital signs
* Laboratory measurements
* Clinical measurements used in domain construction

`Age` was excluded from forward filling because it did not contain missing values.

Forward filling was performed separately for the training, validation, and test datasets.

Values were never propagated between different patients.

After forward filling, missingness was substantially reduced for many frequently measured variables.

However, considerable missingness remained for laboratory variables that were infrequently measured.

---

## 5. Missing-Value Strategy

Missing values were intentionally **not replaced using global mean or median imputation**.

After forward filling, many clinical domain scores still contained missing values.

The final modeling approach used XGBoost, which supports missing numerical values natively.

Therefore, the remaining missing values were preserved rather than artificially replacing potentially meaningful missingness patterns.

This approach also avoided introducing potentially misleading values into highly incomplete laboratory variables.

---

## 6. Clinical Reference Ranges

To transform the raw clinical measurements into comparable clinical-domain information, heuristic reference ranges were defined for the clinical variables.

Examples included:

| Feature    | Lower | Upper |
| ---------- | ----: | ----: |
| HR         |    60 |   100 |
| SBP        |    90 |   140 |
| MAP        |    65 |   100 |
| DBP        |    60 |    80 |
| Resp       |    12 |    20 |
| O2Sat      |    94 |   100 |
| Temp       |  36.0 |  37.5 |
| pH         |  7.35 |  7.45 |
| PaCO2      |    35 |    45 |
| Lactate    |   0.5 |   2.0 |
| Glucose    |    70 |   140 |
| BUN        |     7 |    20 |
| Creatinine |   0.6 |   1.3 |
| WBC        |     4 |    11 |
| Platelets  |   150 |   450 |
| Hgb        |    12 |    17 |

The complete reference ranges were defined for all 34 clinical variables.

These ranges were used as feature-engineering references and were **not treated as validated diagnostic thresholds**.

---

## 7. Clinical Deviation Transformation

For each clinical measurement, the distance from the reference range was calculated.

Values within the reference interval received a deviation score of zero.

Values below or above the interval received a positive deviation proportional to their distance outside the range.

This transformed the raw measurements into a common representation of deviation from the selected reference range.

The resulting deviations were then compressed using:

`log1p(deviation)`

This reduced the influence of extremely large deviations while preserving the ordering of abnormality.

---

## 8. Clinical Domain Construction

The 34 clinical variables were grouped into five broader clinical domains.

### Hemodynamic

* HR
* SBP
* MAP
* DBP

### Respiratory

* Resp
* O2Sat
* Temp
* EtCO2
* FiO2
* pH
* PaCO2
* SaO2

### Renal/Metabolic

* BaseExcess
* HCO3
* BUN
* Creatinine
* Calcium
* Chloride
* Magnesium
* Phosphate
* Potassium
* Glucose
* Lactate

### Inflammatory/Hematological

* WBC
* Platelets
* Hgb
* Hct
* Fibrinogen

### Hepatic/Coagulation

* AST
* Alkalinephos
* Bilirubin_total
* Bilirubin_direct
* PTT
* TroponinI

All 34 clinical variables were assigned to exactly one domain.

---

## 9. Clinical Feature Weighting

Within each domain, individual variables were assigned weights based on three components:

1. Clinical importance prior
2. Training-set evidence/effect size
3. Feature availability

The weighting approach was designed to prevent frequently missing variables from dominating the domain scores while still allowing clinically informative variables to contribute more strongly.

The resulting weights were normalized within each clinical domain.

These weights were project-specific feature-engineering choices and were **not intended to represent validated clinical scoring systems**.

---

## 10. Domain Score Calculation

For each observation, the weighted deviations of the available clinical variables were combined to produce one score for each domain.

The domain score was calculated using the weighted contribution of available features.

If some measurements were missing, the score was normalized by the total weight of the available variables.

If all variables belonging to a domain were missing, the domain score remained missing.

This resulted in five compact clinical features:

* Hemodynamic
* Respiratory
* Renal_Metabolic
* Inflammatory_Hematological
* Hepatic_Coagulation

---

## 11. Domain Missingness

After domain aggregation, the remaining training-set missingness was substantially lower than the original raw laboratory missingness.

The domain-level missingness was:

| Domain                     | Missing |
| -------------------------- | ------: |
| Hemodynamic                |   2.25% |
| Respiratory                |   2.24% |
| Renal_Metabolic            |  12.18% |
| Inflammatory_Hematological |  19.03% |
| Hepatic_Coagulation        |  36.64% |

No observation had all five clinical domain scores missing.

This allowed the model to retain missingness information without requiring artificial imputation.

---

## 12. Domain Correlation Analysis

Correlation analysis was performed between the five domain scores.

The correlations were relatively modest.

Examples included:

* Hemodynamic–Respiratory: **0.123**
* Hemodynamic–Renal/Metabolic: **0.075**
* Respiratory–Renal/Metabolic: **0.107**
* Renal/Metabolic–Inflammatory/Hematological: **0.174**
* Renal/Metabolic–Hepatic/Coagulation: **0.083**

Because the domain scores were not strongly correlated with one another, additional iterative or KNN-based imputation was not considered necessary.

---

## 13. Temporal and Demographic Features

In addition to the five clinical domain scores, the following features were retained:

### Temporal Features

* `Hour`
* `ICULOS`
* `HospAdmTime`

### Demographic Features

* `Gender`

### ICU Feature

* `ICU_Unit`

Temporal variables were retained because exploratory analysis and later modeling experiments showed that they contained substantial predictive information.

---

## 14. Unit1 and Unit2 Reduction

`Unit1` and `Unit2` were examined for redundancy.

A cross-tabulation showed that they were complementary:

* Unit1 = 0 corresponded to Unit2 = 1
* Unit1 = 1 corresponded to Unit2 = 0
* Missing Unit1 corresponded to missing Unit2

Therefore, keeping both variables would introduce redundant information.

A single feature, `ICU_Unit`, was created from `Unit1`, and the original `Unit1` and `Unit2` variables were removed.

Missing ICU-unit information was encoded as **-1** rather than assigning it to either ICU category.

---

## 15. Final Feature Set

Dimensionality reduction resulted in a final model feature set of **10 features**:

1. Hemodynamic
2. Respiratory
3. Renal_Metabolic
4. Inflammatory_Hematological
5. Hepatic_Coagulation
6. Hour
7. ICULOS
8. HospAdmTime
9. Gender
10. ICU_Unit

This reduced the original 43-column working dataset to a compact set of clinically grouped, temporal, demographic, and ICU-related predictors.

`Patient_ID` was retained in the processed CSV files only for patient traceability and was **not used as a model feature**.

`SepsisLabel` remained the target variable.

---

## 16. Final Processed Dataset

The final processed datasets contain:

* 10 model features
* 1 target variable
* 1 patient identifier

Therefore, each exported dataset contains **12 columns**.

The model input matrices contain only the 10 selected features.

### Training

* Observations: **1,086,991**
* Features: **10**

### Validation

* Observations: **155,226**
* Features: **10**

### Test

* Observations: **309,993**
* Features: **10**

---

## 17. Important Data-Quality Considerations

During preprocessing, two unit-related observations were identified:

* `FiO2` values appeared to be stored as fractions such as 0.21, 0.35, 0.50, and 1.0.
* `Lactate` values appeared consistent with a mmol/L-type scale despite a discrepancy with the dataset description.

These observations were documented rather than silently transforming the raw values.

The clinical reference ranges were therefore treated as project-specific feature-engineering references rather than definitive clinical thresholds.

---

## 18. Preprocessing Conclusion

The preprocessing pipeline transformed a high-dimensional and highly incomplete clinical time-series dataset into a compact 10-feature representation.

The key methodological decisions were:

1. Patient-level train/validation/test splitting
2. Patient-wise forward filling
3. Preservation of remaining missing values
4. Reference-range deviation transformation
5. Logarithmic compression of deviations
6. Clinical domain aggregation
7. Evidence-, availability-, and clinical-prior-based weighting
8. Reduction of redundant ICU variables
9. Retention of important temporal variables
10. Exclusion of `Patient_ID` from model inputs

The resulting datasets provided a compact representation suitable for the subsequent XGBoost modeling stage while preserving important clinical, temporal, and missingness information.
