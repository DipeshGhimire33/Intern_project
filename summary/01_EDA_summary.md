# Exploratory Data Analysis Summary

## 1. Dataset Overview

The dataset used for this project is a clinical time-series dataset for sepsis prediction derived from the PhysioNet/CinC Challenge 2019 dataset.

The original dataset contained:

* **1,552,210 observations**
* **44 columns**

After removing the unnamed index column, the working dataset contained:

* **1,552,210 observations**
* **43 meaningful columns**

The dataset contains physiological measurements, laboratory measurements, demographic information, ICU-related temporal information, patient identifiers, and the sepsis prediction target.

---

## 2. Feature Groups

The available variables were organized into the following groups:

### Vital Signs

* HR
* O2Sat
* Temp
* SBP
* MAP
* DBP
* Resp
* EtCO2

### Laboratory Measurements

* BaseExcess
* HCO3
* FiO2
* pH
* PaCO2
* SaO2
* AST
* BUN
* Alkalinephos
* Calcium
* Chloride
* Creatinine
* Bilirubin_direct
* Glucose
* Lactate
* Magnesium
* Phosphate
* Potassium
* Bilirubin_total
* TroponinI
* Hct
* Hgb
* PTT
* WBC
* Fibrinogen
* Platelets

### Demographic and ICU Variables

* Age
* Gender
* Unit1
* Unit2
* HospAdmTime
* ICULOS

### Target

* SepsisLabel

### Patient Identifier

* Patient_ID

---

## 3. Missing Value Analysis

The dataset contained substantial missingness, particularly among laboratory measurements.

Examples of highly incomplete variables included:

* Bilirubin_direct: **99.81%**
* Fibrinogen: **99.34%**
* TroponinI: **99.05%**
* Bilirubin_total: **98.51%**
* Alkalinephos: **98.39%**
* AST: **98.38%**
* Lactate: **97.33%**
* PTT: **97.06%**

Vital signs were generally more complete.

For example:

* HR: **9.88%**
* MAP: **12.45%**
* O2Sat: **13.06%**
* SBP: **14.58%**
* Resp: **15.35%**

The large difference in missingness between routine vital signs and laboratory tests was an important consideration for the later preprocessing strategy.

---

## 4. Duplicate Analysis

No duplicate observations were identified in the dataset.

Therefore, no duplicate rows were removed during preprocessing.

---

## 5. Target Variable Analysis

The target variable `SepsisLabel` was highly imbalanced.

| Class      |     Count | Percentage |
| ---------- | --------: | ---------: |
| Non-Sepsis | 1,524,294 |     98.20% |
| Sepsis     |    27,916 |      1.80% |

Only approximately **1.8% of observations** were labeled as sepsis.

The `SepsisLabel` represents the six-hour pre-sepsis prediction window defined by the dataset.

Because of the severe class imbalance, accuracy alone would not be an appropriate primary evaluation metric.

---

## 6. Patient-Level Analysis

The dataset contained:

* **40,336 unique patients**
* Mean observations per patient: **38.48**
* Median observations per patient: **38**
* Minimum observations: **8**
* Maximum observations: **336**

At the patient level:

* Sepsis-positive patients: **2,932**
* Non-sepsis patients: **37,404**

Approximately **7.27% of patients** experienced at least one sepsis-positive observation.

The large number of repeated observations per patient demonstrated that individual rows could not be treated as completely independent observations.

This finding was important for model development because random row-level splitting could result in patient information appearing across training and evaluation sets. Therefore, patient-level splitting was used during preprocessing.

---

## 7. Numerical Feature Analysis

Summary statistics were examined for the major physiological and laboratory variables.

Several variables showed substantial variation and skewness, particularly laboratory measurements such as:

* Creatinine
* BUN
* Lactate
* Glucose
* WBC

The distributions indicated that simple assumptions of normally distributed clinical variables would not be appropriate.

---

## 8. Outlier Analysis

IQR-based outlier analysis was performed on selected numerical features.

The proportion of observations identified as statistical outliers included:

| Feature    | Approx. Outlier Percentage |
| ---------- | -------------------------: |
| Lactate    |                      8.51% |
| Creatinine |                     11.77% |
| Glucose    |                      5.63% |
| BUN        |                      8.24% |
| WBC        |                      3.49% |
| Platelets  |                      3.17% |
| Resp       |                      2.12% |
| O2Sat      |                      1.84% |

No observations were removed solely because they were statistical outliers.

In clinical data, extreme values can represent genuine patient conditions rather than measurement errors. Therefore, the outlier analysis was used for understanding the data rather than automatically deleting observations.

---

## 9. Correlation Analysis

Several strong correlations were identified between clinical variables.

Notable positive correlations included:

* Bilirubin_direct and Bilirubin_total: **0.964**
* Hct and Hgb: **0.952**
* BaseExcess and HCO3: **0.857**
* MAP and DBP: **0.852**
* SBP and MAP: **0.780**
* BaseExcess and pH: **0.651**
* BUN and Creatinine: **0.624**

Notable negative correlations included:

* Lactate and BaseExcess: **-0.448**
* PaCO2 and pH: **-0.437**
* Lactate and HCO3: **-0.406**
* Phosphate and pH: **-0.396**
* Lactate and pH: **-0.370**

These relationships demonstrated substantial redundancy among some raw clinical variables.

This supported the later dimensionality-reduction strategy of aggregating related measurements into clinical domain scores rather than retaining all raw measurements as independent model features.

---

## 10. Feature Behaviour by Sepsis Status

Several clinical variables showed different central tendencies between non-sepsis and sepsis observations.

Examples of mean values were:

| Feature    | Non-Sepsis Mean | Sepsis Mean |
| ---------- | --------------: | ----------: |
| HR         |           84.47 |       90.79 |
| O2Sat      |           97.20 |       96.99 |
| Temp       |           36.97 |       37.25 |
| SBP        |          123.79 |      121.45 |
| MAP        |           82.44 |       80.17 |
| DBP        |           63.86 |       62.02 |
| Resp       |           18.69 |       20.46 |
| Glucose    |          136.87 |      140.03 |
| Lactate    |            2.64 |        2.73 |
| WBC        |           11.40 |       13.33 |
| Creatinine |            1.50 |        1.85 |
| BUN        |           23.75 |       30.77 |

The distributions showed considerable overlap between the two classes.

Therefore, individual variables did not provide a clear separation between sepsis and non-sepsis observations.

---

## 11. Temporal Analysis

The temporal variables `Hour` and `ICULOS` represented the patient's position in the ICU time series.

Observed ranges included approximately:

* Hour: **0–335**
* ICULOS: **1–336**

The number of observations decreased at later ICU hours because patients had different lengths of ICU stay.

The observed sepsis rate varied across ICU time, but later observations also had smaller sample sizes and different patient composition.

Therefore, the temporal patterns were treated as predictive patterns rather than evidence of a causal relationship between ICU duration and sepsis.

---

## 12. Important EDA Findings

The main findings from the exploratory analysis were:

1. The dataset is highly imbalanced, with approximately **1.8% sepsis observations**.
2. There are **40,336 unique patients**, with multiple observations per patient.
3. Laboratory variables contain substantial missingness.
4. Vital signs are generally more complete than laboratory variables.
5. Several clinical variables are strongly correlated.
6. Statistical outliers are present but may represent genuine clinical observations.
7. Sepsis and non-sepsis observations show overlapping distributions for individual clinical variables.
8. Temporal information contains potentially useful predictive patterns.
9. Patient-level splitting is necessary to reduce the risk of patient-level data leakage.
10. The strong correlations and high dimensionality of the raw clinical variables motivated a dimensionality-reduction approach based on clinically grouped domain scores.

---

## 13. EDA Conclusion

The exploratory analysis showed that the dataset presents several challenges for machine-learning development, particularly severe class imbalance, extensive missing laboratory measurements, repeated observations from the same patients, correlated clinical variables, and overlapping feature distributions between sepsis and non-sepsis observations.

These findings motivated the preprocessing strategy used in the next stage of the project, including patient-level dataset splitting, patient-wise forward filling, clinical deviation features, clinical domain aggregation, dimensionality reduction, and preservation of missing values for models capable of handling them natively.