# Sepsis Prediction Using Machine Learning

## Project Overview

This project focuses on developing a machine learning pipeline for **early prediction of sepsis using clinical data**.

The project uses patient-level clinical observations from the **PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data** dataset. The workflow covers exploratory data analysis, patient-level data splitting, missing-value handling, clinical feature engineering, machine learning model development, baseline comparison, threshold selection, and final evaluation.

The main objective is to investigate whether aggregated clinical and temporal information can be used to identify patients at increased risk of sepsis.

> **Important:** This is an educational/research project and is not intended for clinical diagnosis, treatment decisions, or deployment in healthcare settings.

---

## Objectives

* Explore the structure and characteristics of the sepsis dataset.
* Analyze class imbalance, missing values, correlations, and temporal patterns.
* Prevent patient-level data leakage through patient-wise train/validation/test splitting.
* Apply patient-wise forward filling to preserve the temporal structure of clinical observations.
* Engineer clinically meaningful domain-level features.
* Develop an XGBoost-based sepsis prediction model.
* Compare XGBoost with several baseline machine learning models.
* Evaluate models using metrics appropriate for highly imbalanced classification.
* Select decision thresholds using the validation set.
* Perform final evaluation on the held-out test set.
* Analyze model feature importance and interpretability.
* Document the complete methodology and findings for reproducibility and deeper understanding.

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Joblib
* Jupyter Notebook

---

## Project Structure

```text
sepsis-prediction/
├── models/
│   ├── xgboost_sepsis_model.pkl
│   ├── logistic_regression_baseline.pkl
│   ├── random_forest_baseline.pkl
│   ├── hist_gradient_boosting_baseline.pkl
│   └── linear_svm_baseline.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_preprocessing.ipynb
│   ├── 03_modeling_xgboost.ipynb
│   ├── 04_modeling_baselines.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_final_evaluation.ipynb
├── results/
│   ├── test_model_comparison.csv
│   ├── validation_model_comparison.csv
│   └── xgboost_classification_report.csv
├── summary/
│   ├── 01_EDA_summary.md
│   ├── 02_Data_preprocessing.md
│   ├── 03_modelling_Xgboost_summary.md
│   ├── 04_model_baseline_summary.md
│   ├── 05_model_comparison_summary.md
│   └── 06_final_evaluation_summary.md
├── requirements.txt
├── README.md
└── .gitignore
```

> The raw and processed datasets are intentionally excluded from the repository because of their large file sizes. They are used locally during the notebook workflow.

---

# Dataset

The project is based on the **PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data** dataset.

### Sources

* **PhysioNet/CinC Challenge 2019** — official challenge dataset and documentation.
* **Kaggle — Prediction of Sepsis** — dataset source used for this project.

The original dataset contains clinical observations collected over time for patients admitted to intensive care units.

### Dataset Characteristics

After removing the unnamed/index column:

* **Rows:** 1,552,210
* **Columns:** 43
* **Unique patients:** 40,336
* **Non-sepsis observations:** 1,524,294
* **Sepsis observations:** 27,916
* **Sepsis observation rate:** approximately 1.80%

At the patient level:

* **Sepsis-positive patients:** 2,932
* **Sepsis-negative patients:** 37,404
* **Patient-level positive rate:** approximately 7.27%

The dataset is therefore highly imbalanced, making metrics such as **PR-AUC, precision, recall, and F1-score** particularly important.

### Dataset Availability

The raw and processed dataset files are **not included in this GitHub repository** because of their large file sizes.

The original dataset can be obtained from the PhysioNet/CinC Challenge 2019 and Kaggle sources.

The local workflow uses:

```text
data/raw/raw_Dataset.csv

data/processed/train_processed.csv
data/processed/val_processed.csv
data/processed/test_processed.csv
```

These files are excluded from version control through `.gitignore`.

The notebooks document the preprocessing workflow used to generate the processed datasets.

---

# Exploratory Data Analysis

The exploratory analysis examined:

* Dataset dimensions and feature types
* Target distribution
* Patient-level statistics
* Missing-value patterns
* Duplicate records
* Feature distributions
* Outliers
* Correlations between clinical variables
* Temporal patterns across ICU hours
* Differences between observation-level and patient-level class distributions

### Important EDA Findings

The dataset contains substantial missingness, particularly among laboratory measurements.

Examples include:

* Bilirubin_direct: approximately 99.81% missing
* Fibrinogen: approximately 99.34% missing
* TroponinI: approximately 99.05% missing
* Lactate: approximately 97.33% missing
* WBC: approximately 93.59% missing

Missingness is therefore an important characteristic of the dataset rather than a simple preprocessing inconvenience.

No duplicate records were identified.

Several clinically related variables were strongly correlated, including:

* Bilirubin_direct and Bilirubin_total
* Hct and Hgb
* BaseExcess and HCO3
* MAP and DBP
* SBP and MAP

The raw observation count also decreases over ICU time because patients have different ICU stay lengths.

### Detailed EDA Documentation

For a detailed explanation of the exploratory analysis, findings, and interpretation, see:

[01_EDA_summary.md](summary/01_EDA_summary.md)

---

# Patient-Level Data Splitting

To reduce the risk of patient-level data leakage, the dataset was divided using **Patient_ID rather than individual observations**.

The approximate split was:

* **70% training**
* **10% validation**
* **20% testing**

This ensures that observations belonging to the same patient do not appear across multiple dataset partitions.

Target distribution:

| Split      | Non-Sepsis | Sepsis |
| ---------- | ---------: | -----: |
| Train      |  1,067,723 | 19,268 |
| Validation |    152,450 |  2,776 |
| Test       |    304,121 |  5,872 |

The validation set was used for model comparison and threshold selection.

The test set was reserved for final performance reporting.

---

# Data Preprocessing

## Patient-Wise Forward Filling

Clinical observations are collected sequentially over time.

Therefore, missing clinical measurements were handled using **forward filling within each patient**:

```python
data.groupby("Patient_ID")[features].ffill()
```

This allows a previously observed clinical value to remain available until a newer measurement is recorded.

Forward filling was performed separately within each dataset split to avoid cross-patient and cross-split information leakage.

`Age` was excluded from forward filling because it was already complete.

### Remaining Missingness

After forward filling, some laboratory variables continued to have substantial missingness.

This was expected because many laboratory tests are not performed at every ICU time point.

Rather than aggressively imputing these values, the project preserved missingness for the tree-based XGBoost model, which can handle missing values natively.

Baseline models requiring complete input used median imputation.

---

# Clinical Domain Feature Engineering

The original PhysioNet/CinC documentation defines the individual clinical variables, but it does **not prescribe the five project-specific physiological domains used here**.

The five domains were created as a **project-specific feature-engineering strategy** based on the physiological or laboratory meaning of the variables.

The domain assignments are therefore not intended to represent formal diagnostic categories or validated clinical scoring systems.

| Domain                                   | Features                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Hemodynamic**                          | HR, SBP, MAP, DBP                                                                                       |
| **Respiratory**                          | Resp, O2Sat, Temp, EtCO2, FiO2, pH, PaCO2, SaO2                                                         |
| **Renal / Metabolic**                    | BaseExcess, HCO3, BUN, Creatinine, Calcium, Chloride, Magnesium, Phosphate, Potassium, Glucose, Lactate |
| **Inflammatory / Hematological**         | WBC, Platelets, Hgb, Hct, Fibrinogen                                                                    |
| **Hepatic / Coagulation / Organ Injury** | AST, Alkalinephos, Bilirubin_total, Bilirubin_direct, PTT, TroponinI                                    |

A few variables require additional interpretation:

* **Temperature** is a general vital sign rather than a strictly respiratory measurement.
* **TroponinI** is primarily a marker of cardiac injury and is included in the final broad laboratory/organ-injury domain rather than being interpreted as a hepatic marker.

### Domain References

The clinical meaning of the variables was informed by the following references:

* [PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis](https://physionet.org/content/challenge-2019/1.0.0/)
* [MedlinePlus — Complete Blood Count (CBC)](https://medlineplus.gov/lab-tests/complete-blood-count-cbc/)
* [MedlinePlus — Kidney Function Tests](https://medlineplus.gov/ency/article/003435.htm)
* [MedlinePlus — Basic Metabolic Panel (BMP)](https://medlineplus.gov/lab-tests/basic-metabolic-panel-bmp/)
* [Merck Manual — Laboratory Tests of the Liver and Gallbladder](https://www.merckmanuals.com/professional/hepatic-and-biliary-disorders/testing-for-hepatic-and-biliary-disorders/laboratory-tests-of-the-liver-and-gallbladder)

The domain assignments themselves are **project-specific feature-engineering categories**, not official PhysioNet categories or validated clinical classifications.

---

# Domain-Level Deviation Scores

To summarize heterogeneous clinical measurements, heuristic reference ranges were used to calculate how far an observed value deviated from its reference interval.

For an observed value:

* Values inside the reference interval received a deviation score of 0.
* Values below or above the interval received a normalized deviation.
* `log1p` transformation was subsequently applied to reduce the effect of extreme deviations.

These reference ranges were used **only for feature engineering** and should not be interpreted as validated sepsis diagnostic thresholds.

The resulting feature values were then aggregated into five domain-level scores using weighted averages.

The weighting incorporated:

* Project-specific clinical importance
* Observed association with the target in the training data
* Feature availability

The resulting domain scores were:

* Hemodynamic
* Respiratory
* Renal_Metabolic
* Inflammatory_Hematological
* Hepatic_Coagulation

This approach reduced the original clinical feature space while preserving broader physiological information.

---

# Temporal and Demographic Features

In addition to the five clinical domain scores, the final feature set included selected temporal and demographic information.

### Temporal Features

* `Hour`
* `ICULOS`
* `HospAdmTime`

### Demographic / ICU Features

* `Gender`
* `ICU_Unit`

`Unit1` and `Unit2` were found to be complementary indicators of ICU unit assignment, so they were represented using a single `ICU_Unit` feature.

---

# Final Feature Set

The final model used **10 features**:

```text
Hemodynamic
Respiratory
Renal_Metabolic
Inflammatory_Hematological
Hepatic_Coagulation
Hour
ICULOS
HospAdmTime
Gender
ICU_Unit
```

The final feature design combines:

1. Aggregated physiological information
2. Temporal progression
3. Demographic information
4. ICU context

---

# XGBoost Model

The primary model was an **XGBoost binary classification model**.

Configuration:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=55.4143,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

Because the target was highly imbalanced, `scale_pos_weight` was used to increase the importance of the minority class during training.

The validation learning curve indicated that the best validation PR-AUC occurred around the first 48 boosting iterations.

For final evaluation, the saved 300-tree model was therefore evaluated using:

```python
iteration_range=(0, 48)
```

---

# Baseline Models

Four baseline models were evaluated against XGBoost:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* Linear SVM

Median imputation was used for models that could not directly handle missing values.

The comparison was performed using the same patient-level train/validation/test framework.

---

# Evaluation Metrics

Because sepsis observations represent a small minority of the dataset, accuracy alone is not sufficient for evaluating model performance.

The project therefore focuses on:

* **PR-AUC:** important for imbalanced classification
* **ROC-AUC:** measures ranking/discrimination ability
* **Precision:** proportion of predicted positives that were actually positive
* **Recall:** proportion of actual positives detected
* **F1-score:** harmonic mean of precision and recall
* **Accuracy:** overall classification accuracy

Decision thresholds were selected using the validation set rather than the test set.

---

# Final Test Results

The following results were obtained on the held-out test set.

| Model                | Accuracy | Precision |   Recall | F1-Score |  ROC-AUC |   PR-AUC |
| -------------------- | -------: | --------: | -------: | -------: | -------: | -------: |
| Logistic Regression  | 0.947205 |  0.106081 | 0.240634 | 0.147249 | 0.743265 | 0.073414 |
| Random Forest        | 0.955896 |  0.139223 | 0.256301 | 0.180434 | 0.787584 | 0.091238 |
| XGBoost              | 0.955805 |  0.135839 | 0.248638 | 0.175692 | 0.791266 | 0.092856 |
| HistGradientBoosting | 0.954060 |  0.128011 | 0.245232 | 0.168214 | 0.786422 | 0.088892 |
| Linear SVM           | 0.775450 |  0.050508 | 0.609843 | 0.093290 | 0.744026 | 0.071523 |

### Interpretation

XGBoost achieved the highest **test PR-AUC (0.092856)** and **ROC-AUC (0.791266)** among the evaluated models.

Random Forest achieved a slightly higher **F1-score (0.180434)** and **recall (0.256301)** at its selected threshold.

These differences demonstrate why multiple evaluation metrics are reported rather than selecting a model using a single metric.

Because the dataset is highly imbalanced, PR-AUC provides particularly useful information about positive-class performance.

---

# Decision Threshold

The default classification threshold of 0.50 was not assumed to be optimal.

Thresholds were evaluated on the validation set, and a threshold was selected for each model based on validation performance.

For the final XGBoost model:

```text
Validation-selected threshold = 0.80
```

The threshold was selected **only using validation data** and then applied unchanged to the held-out test set.

---

# Final XGBoost Evaluation

Using the first 48 boosting iterations and a classification threshold of 0.80, the final XGBoost test performance was:

| Metric    |    Score |
| --------- | -------: |
| Accuracy  | 0.955805 |
| Precision | 0.135839 |
| Recall    | 0.248638 |
| F1-Score  | 0.175692 |
| ROC-AUC   | 0.791266 |
| PR-AUC    | 0.092856 |

### Confusion Matrix

```text
[[294833   9288]
 [  4412   1460]]
```

Where:

* True Negatives = 294,833
* False Positives = 9,288
* False Negatives = 4,412
* True Positives = 1,460

---

# Model Interpretability

XGBoost feature importance was examined to understand which engineered features contributed most strongly to the model.

| Feature                    | Importance |
| -------------------------- | ---------: |
| ICULOS                     |   0.307865 |
| Respiratory                |   0.134946 |
| Hour                       |   0.107581 |
| ICU_Unit                   |   0.079999 |
| HospAdmTime                |   0.077891 |
| Renal_Metabolic            |   0.066904 |
| Gender                     |   0.062616 |
| Inflammatory_Hematological |   0.061798 |
| Hepatic_Coagulation        |   0.053556 |
| Hemodynamic                |   0.046843 |

`ICULOS` had the largest built-in feature importance, followed by the Respiratory domain and Hour.

Permutation importance and SHAP analysis were also used to provide additional interpretability.

Feature importance should be interpreted as **model behavior**, not as evidence that a feature causally produces sepsis.

---

# Key Findings

* The dataset is highly imbalanced at the observation level.
* Patient-level splitting is important to reduce leakage from repeated observations of the same patient.
* Clinical measurements contain substantial missingness, particularly among laboratory tests.
* Patient-wise forward filling preserves temporal information while avoiding cross-patient filling.
* Aggregating clinical measurements into physiological domains provides a compact feature representation.
* Temporal features, particularly `ICULOS` and `Hour`, contributed substantially to model predictions.
* XGBoost achieved a test PR-AUC of **0.092856** and ROC-AUC of **0.791266**.
* Random Forest achieved a slightly higher F1-score and recall at its selected threshold.
* No single metric completely describes performance on this highly imbalanced problem.

---

# Limitations

Several limitations should be considered:

* The project uses a historical benchmark dataset rather than prospectively collected clinical data.
* The dataset contains substantial missingness.
* Forward filling assumes that a previously observed value remains informative until a new measurement is available.
* The clinical reference ranges used for deviation features are heuristic feature-engineering ranges and are not validated sepsis thresholds.
* The five clinical domains are project-defined groupings rather than formal clinical scoring systems.
* The dataset contains repeated observations per patient, so temporal dependence remains important.
* Model performance may not generalize to other hospitals, populations, or clinical workflows.
* The model has not undergone external validation.
* The project is not intended for clinical deployment or patient-level medical decision-making.

---

# Future Improvements

Potential future work includes:

* Hyperparameter optimization using systematic search.
* More extensive temporal feature engineering.
* Rolling-window and trend-based features.
* Missingness indicator features.
* Patient-level sequential models.
* LSTM, GRU, or Transformer-based approaches.
* More extensive calibration analysis.
* External validation using an independent dataset.
* Cost-sensitive threshold optimization based on an explicit clinical objective.
* Comparison with additional boosting algorithms.
* More detailed subgroup and fairness analysis.

---

# Notebook Workflow

The project is organized into six main notebooks.

### 01 — Exploratory Data Analysis

`notebooks/01_EDA.ipynb`

Performs dataset exploration, class distribution analysis, missingness analysis, patient-level statistics, correlations, outlier analysis, and temporal analysis.

**Detailed documentation:**

[01_EDA_summary.md](summary/01_EDA_summary.md)

---

### 02 — Data Preprocessing

`notebooks/02_Data_preprocessing.ipynb`

Covers patient-level splitting, patient-wise forward filling, clinical deviation features, domain-level aggregation, ICU unit representation, and construction of the final feature set.

**Detailed documentation:**

[02_Data_preprocessing.md](summary/02_Data_preprocessing.md)

---

### 03 — XGBoost Modeling

`notebooks/03_modeling_xgboost.ipynb`

Develops the XGBoost model, handles class imbalance, evaluates validation performance, examines learning behavior, performs threshold analysis, and investigates feature importance.

**Detailed documentation:**

[03_modelling_Xgboost_summary.md](summary/03_modelling_Xgboost_summary.md)

---

### 04 — Baseline Models

`notebooks/04_modeling_baselines.ipynb`

Trains and evaluates Logistic Regression, Random Forest, HistGradientBoosting, and Linear SVM baseline models.

**Detailed documentation:**

[04_model_baseline_summary.md](summary/04_model_baseline_summary.md)

---

### 05 — Model Comparison

`notebooks/05_model_comparison.ipynb`

Compares all evaluated models using validation and held-out test metrics and applies validation-selected thresholds.

**Detailed documentation:**

[05_model_comparison_summary.md](summary/05_model_comparison_summary.md)

---

### 06 — Final Evaluation

`notebooks/06_final_evaluation.ipynb`

Performs the final XGBoost evaluation using the validation-selected configuration and held-out test set. Includes the confusion matrix, PR curve, ROC curve, classification report, feature importance, interpretation, limitations, and conclusion.

**Detailed documentation:**

[06_final_evaluation_summary.md](summary/06_final_evaluation_summary.md)

---

# Detailed Project Summaries

For readers who want a deeper understanding of the methodology and reasoning behind each stage, the repository includes a dedicated summary for every notebook.

| Notebook                      | Detailed Summary                                                           |
| ----------------------------- | -------------------------------------------------------------------------- |
| `01_EDA.ipynb`                | [01_EDA_summary.md](summary/01_EDA_summary.md)                             |
| `02_Data_preprocessing.ipynb` | [02_Data_preprocessing.md](summary/02_Data_preprocessing.md)               |
| `03_modeling_xgboost.ipynb`   | [03_modelling_Xgboost_summary.md](summary/03_modelling_Xgboost_summary.md) |
| `04_modeling_baselines.ipynb` | [04_model_baseline_summary.md](summary/04_model_baseline_summary.md)       |
| `05_model_comparison.ipynb`   | [05_model_comparison_summary.md](summary/05_model_comparison_summary.md)   |
| `06_final_evaluation.ipynb`   | [06_final_evaluation_summary.md](summary/06_final_evaluation_summary.md)   |

> **For in-depth understanding:** The summary files explain the methodology, important implementation decisions, results, interpretation, and limitations of each notebook. The notebooks contain the corresponding code and analysis.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/DipeshGhimire33/Intern_project.git
cd Intern_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

---

# Usage

The notebooks should generally be executed in the following order:

```text
01_EDA.ipynb
      ↓
02_Data_preprocessing.ipynb
      ↓
03_modeling_xgboost.ipynb
      ↓
04_modeling_baselines.ipynb
      ↓
05_model_comparison.ipynb
      ↓
06_final_evaluation.ipynb
```

The required dataset files are not included in the repository and must be obtained separately from the dataset sources.

---

# Results and Reproducibility

The repository contains:

* Trained model files in `models/`
* Validation comparison results in `results/validation_model_comparison.csv`
* Test comparison results in `results/test_model_comparison.csv`
* XGBoost classification report in `results/xgboost_classification_report.csv`
* Detailed notebook summaries in `summary/`
* Complete analysis notebooks in `notebooks/`

The datasets themselves are excluded from version control because of their large size. The preprocessing notebook documents how the local processed datasets were generated.

---

# Author

**Dipesh Ghimire**

Machine Learning / Data Science Internship Project

**GitHub:** DipeshGhimire33/Intern_project

**LinkedIn:** Dipesh Ghimire

---

# License

This project is intended for educational and research purposes.

---

# Disclaimer

This project is **not a medical device, diagnostic system, or clinical decision-support tool**.

The predictions and analyses presented here are intended only for educational and research purposes and should not be used to diagnose sepsis, guide treatment, or make clinical decisions.
