# 🩺 Sepsis Prediction Using Machine Learning

## 📌 Project Overview

This project was developed as part of my **Data Science and Machine Learning Internship**.

The objective of this project is to develop and evaluate a **machine learning-based approach for predicting sepsis from longitudinal clinical patient data**. The project follows a complete machine learning workflow, including data exploration, patient-level data splitting, preprocessing, feature engineering, model development, model comparison, and final evaluation.

The dataset is based on the **PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data** and contains physiological measurements, laboratory measurements, demographic information, and temporal variables collected over patients' ICU stays.

The project focuses on predicting the dataset's `SepsisLabel`, which identifies observations within the dataset-defined sepsis prediction window.

> **⚠️ Important:** This project is intended for educational and research purposes. The resulting model is **not a medical diagnostic system** and must not be used to make clinical decisions.

---

## 🎯 Objectives

The main objectives of this project are:

* Understand and analyze a longitudinal clinical dataset
* Perform exploratory data analysis (EDA)
* Analyze missingness, class imbalance, correlations, and outliers
* Prevent patient-level data leakage during model development
* Develop a clinically informed feature-engineering approach
* Reduce the original feature space into a smaller set of interpretable domain-level features
* Handle missing clinical measurements while preserving information where possible
* Develop an XGBoost-based sepsis prediction model
* Develop traditional machine learning baseline models
* Compare model performance using appropriate classification metrics
* Analyze classification-threshold effects
* Examine model feature importance and interpretability
* Perform final evaluation on a held-out test dataset

---

## 🛠️ Technologies Used

* **Python** — Programming language
* **NumPy** — Numerical computation
* **Pandas** — Data manipulation and analysis
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Scikit-learn** — Machine learning and evaluation
* **XGBoost** — Gradient boosting model
* **SHAP** — Model interpretability
* **Jupyter Notebook** — Development and experimentation
* **Joblib** — Model serialization

---

## 📂 Project Structure

```text
sepsis-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── xgboost_sepsis_model.pkl
│   ├── logistic_regression_baseline.pkl
│   ├── random_forest_baseline.pkl
│   ├── hist_gradient_boosting_baseline.pkl
│   └── linear_svm_baseline.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_modeling_xgboost.ipynb
│   ├── 04_modeling_baselines.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_final_evaluation.ipynb
│
├── results/
│   ├── validation_model_comparison.csv
│   ├── test_model_comparison.csv
│   └── xgboost_classification_report.txt
│
├── summary/
│   ├── 01_eda_summary.md
│   ├── 02_preprocessing_summary.md
│   ├── 03_xgboost_modeling_summary.md
│   ├── 04_baseline_models_summary.md
│   ├── 05_model_comparison_summary.md
│   └── 06_final_evaluation_summary.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The dataset used in this project is based on the:

**PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data**

The Kaggle version used for this project is:

**Kaggle — Prediction of Sepsis**

Dataset:
https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

The dataset contains longitudinal observations collected from patients during their ICU stays.

The original dataset used in this project contains:

* **1,552,210 observations**
* **43 variables after removing the original index column**
* **40,336 unique patients**
* **34 clinical measurements**
* Demographic variables
* ICU-related variables
* Temporal variables
* `SepsisLabel` as the prediction target

### Main types of variables

The clinical measurements include:

* Heart rate
* Oxygen saturation
* Temperature
* Systolic blood pressure
* Mean arterial pressure
* Diastolic blood pressure
* Respiratory rate
* End-tidal CO₂
* Blood gas measurements
* Glucose
* Lactate
* BUN
* Creatinine
* WBC
* Platelets
* Hemoglobin
* Hematocrit
* Liver-related measurements
* Coagulation-related measurements
* Electrolytes

Additional variables include:

* Age
* Gender
* ICU unit information
* Hospital admission time
* ICU length of stay
* Hour
* Patient ID

The target variable is:

```text
SepsisLabel
```

where `1` represents the positive class according to the dataset's sepsis-label definition and `0` represents the negative class.

---

# 🔍 Methodology

## 1. Data Loading and Initial Inspection

The raw dataset was loaded using Pandas and examined for:

* Dataset dimensions
* Data types
* Feature names
* Missing values
* Duplicate records
* Target distribution
* Patient-level structure

The original unnamed/index column was removed before analysis.

---

## 2. Exploratory Data Analysis

EDA was performed in `01_eda.ipynb`.

The analysis included:

* Dataset structure
* Missing-value analysis
* Duplicate analysis
* Target-class distribution
* Patient-level analysis
* Numerical feature distributions
* Outlier analysis
* Correlation analysis
* Feature comparison between sepsis and non-sepsis observations
* Time-based analysis
* Interpretation of important patterns

### Class imbalance

The observation-level target distribution was:

| Class      |     Count | Percentage |
| ---------- | --------: | ---------: |
| Non-Sepsis | 1,524,294 |     98.20% |
| Sepsis     |    27,916 |      1.80% |

This substantial class imbalance made **Precision-Recall AUC (PR-AUC)** an important evaluation metric.

At the patient level, approximately **7.27% of patients** had at least one positive sepsis observation.

---

## 3. Patient-Level Data Splitting

Because each patient contributes multiple observations over time, randomly splitting individual rows could cause information from the same patient to appear in both training and evaluation sets.

To reduce this form of patient-level data leakage, the dataset was split based on unique `Patient_ID`.

The approximate split was:

* **70% training**
* **10% validation**
* **20% testing**

The same patient appears in only one of these datasets.

The target distributions were:

| Dataset    | Non-Sepsis | Sepsis |
| ---------- | ---------: | -----: |
| Training   |  1,067,723 | 19,268 |
| Validation |    152,450 |  2,776 |
| Test       |    304,121 |  5,872 |

---

# 🧹 Data Preprocessing

Preprocessing was performed in `02_data_preprocessing.ipynb`.

## 4. Patient-Wise Forward Filling

Clinical measurements are collected irregularly, meaning that a measurement may not be recorded at every hour.

For suitable clinical variables, missing observations were forward-filled **within each patient**.

The data was first sorted by:

```text
Patient_ID
Hour
```

Forward filling was then performed separately for each patient.

This prevents measurements from one patient from being propagated into another patient's records.

`Age` was excluded from forward filling because it contained no missing values.

---

## 5. Missing Values

After patient-wise forward filling, substantial missingness remained, particularly among laboratory variables.

Instead of aggressively imputing all clinical measurements, the project preserved missing values in the core feature representation.

This was chosen because missingness itself can contain information about which clinical measurements were ordered or available.

The final tree-based modeling approach was able to handle missing values directly.

For baseline models that required complete numerical inputs, median imputation was fitted using the training data and then applied to validation and test data.

---

# 🧠 Feature Engineering

## 6. Clinical Domain Features

Rather than expanding the feature space with a large number of individual derived variables, the project used a **dimensionality-reduction approach**.

The original clinical variables were grouped into five clinical domains:

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

### Renal / Metabolic

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

### Inflammatory / Hematological

* WBC
* Platelets
* Hgb
* Hct
* Fibrinogen

### Hepatic / Coagulation

* AST
* Alkalinephos
* Bilirubin_total
* Bilirubin_direct
* PTT
* TroponinI

---

## 7. Domain-Level Deviation Scores

For the clinical measurements, heuristic reference ranges were used to calculate how far an observed value deviated from a predefined reference interval.

These ranges were used for **feature engineering only** and were not intended to represent validated diagnostic thresholds.

A weighted deviation score was then calculated for each clinical domain.

The weighting incorporated:

* Clinical-importance priors defined for the project
* Feature availability
* Observed deviation from the reference range

The resulting domain scores provided a compact representation of the original clinical variables.

The five resulting clinical features were:

```text
Hemodynamic
Respiratory
Renal_Metabolic
Inflammatory_Hematological
Hepatic_Coagulation
```

---

## 8. Temporal and Demographic Features

Temporal and demographic information was retained because the observations represent a longitudinal ICU process.

The final temporal features were:

```text
Hour
ICULOS
HospAdmTime
```

The final demographic/administrative features were:

```text
Gender
ICU_Unit
```

`Unit1` and `Unit2` contained complementary ICU-unit information and were therefore represented by a single `ICU_Unit` feature.

---

## 9. Final Feature Set

The final model used **10 features**:

```text
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
```

This reduced the original clinical feature space into a smaller and more interpretable representation.

---

# 🤖 Model Development

## 10. XGBoost Model

The primary model was an **XGBoost binary classification model**.

The model used:

* 300 estimators
* Maximum tree depth of 6
* Learning rate of 0.05
* Subsampling
* Feature subsampling
* Class-imbalance weighting
* Histogram-based tree construction

The positive-class weight was approximately:

```text
55.41
```

This reflected the strong imbalance between the sepsis and non-sepsis observations in the training data.

The validation learning curve was used to identify the strongest validation PR-AUC region. The final evaluation reproduced this state using the first **48 trees** of the saved 300-tree model.

---

# 🧪 Baseline Models

Several baseline models were developed for comparison:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* Linear SVM

The baseline models used the same final feature set.

For models that required complete numerical inputs, median imputation was fitted using only the training data.

No StandardScaler was used in the baseline modeling workflow.

---

# 📈 Model Evaluation

Because the dataset is highly imbalanced, accuracy alone does not adequately describe model performance.

The following metrics were used:

### Accuracy

The proportion of all observations classified correctly.

### Precision

The proportion of predicted positive observations that were actually positive.

### Recall

The proportion of actual positive observations that were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures ranking/discrimination performance across classification thresholds.

### PR-AUC

Precision-Recall AUC was used as an important metric because the positive class is relatively uncommon.

### Confusion Matrix

The confusion matrix provides:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

# 📊 Model Comparison Results

The following results were obtained on the held-out test dataset using thresholds selected from the validation data.

| Model                | Accuracy | Precision |   Recall | F1-Score |  ROC-AUC |   PR-AUC |
| -------------------- | -------: | --------: | -------: | -------: | -------: | -------: |
| Logistic Regression  | 0.947205 |  0.106081 | 0.240634 | 0.147249 | 0.743265 | 0.073414 |
| Random Forest        | 0.955896 |  0.139223 | 0.256301 | 0.180434 | 0.787584 | 0.091238 |
| XGBoost              | 0.955805 |  0.135839 | 0.248638 | 0.175692 | 0.791266 | 0.092856 |
| HistGradientBoosting | 0.954060 |  0.128011 | 0.245232 | 0.168214 | 0.786422 | 0.088892 |
| Linear SVM           | 0.775450 |  0.050508 | 0.609843 | 0.093290 | 0.744026 | 0.071523 |

The results demonstrate different precision-recall trade-offs between the models.

XGBoost produced a test **PR-AUC of 0.092856** and **ROC-AUC of 0.791266**.

Random Forest produced a test **F1-score of 0.180434** with precision of 0.139223 and recall of 0.256301.

The Linear SVM produced substantially higher recall at its evaluated threshold, but this came with lower precision and therefore a lower F1-score.

These results illustrate why multiple evaluation metrics are necessary for highly imbalanced classification problems.

---

# 🎯 Classification Threshold

The default classification threshold of 0.50 was not assumed to be optimal.

Threshold analysis was performed using the validation dataset.

For the final XGBoost evaluation, a threshold of:

```text
0.80
```

was selected based on the validation analysis.

At this threshold, the final XGBoost test performance was:

| Metric    |    Value |
| --------- | -------: |
| Accuracy  | 0.955805 |
| Precision | 0.135839 |
| Recall    | 0.248638 |
| F1-Score  | 0.175692 |
| ROC-AUC   | 0.791266 |
| PR-AUC    | 0.092856 |

The threshold was selected using validation data rather than the test set.

---

# 🧮 Final XGBoost Confusion Matrix

The final XGBoost test confusion matrix was:

```text
[[294833   9288]
 [  4412   1460]]
```

Therefore:

|                   | Predicted Non-Sepsis | Predicted Sepsis |
| ----------------- | -------------------: | ---------------: |
| Actual Non-Sepsis |              294,833 |            9,288 |
| Actual Sepsis     |                4,412 |            1,460 |

This corresponds to:

* **True Negatives:** 294,833
* **False Positives:** 9,288
* **False Negatives:** 4,412
* **True Positives:** 1,460

---

# 🔬 Model Interpretability

Model interpretation was investigated using:

* XGBoost built-in feature importance
* Permutation importance
* SHAP analysis

The XGBoost feature importance analysis showed that several temporal and clinical-domain variables contributed substantially to model predictions.

Approximate built-in feature importance:

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

`ICULOS` was the most prominent feature according to the model's built-in importance measure.

Permutation importance also showed strong contributions from `ICULOS`, `Hour`, and the clinical-domain features.

These importance measures describe how the trained model uses the available features. They should **not be interpreted as evidence of causality or clinical importance**.

---

# 💡 Key Findings

The project produced several important observations.

### 1. Strong class imbalance

Only approximately **1.80% of observations** belonged to the positive sepsis class.

This makes accuracy an insufficient standalone evaluation metric.

### 2. High clinical missingness

Many laboratory measurements contained substantial missing values.

Patient-wise forward filling reduced some missingness, but considerable missingness remained in several laboratory-derived domains.

### 3. Patient-level splitting is important

Each patient contributes multiple observations over time.

Therefore, patient-level splitting was used to prevent observations from the same patient from being distributed across training, validation, and test datasets.

### 4. Dimensionality reduction

The original clinical measurements were transformed into five domain-level features:

```text
Hemodynamic
Respiratory
Renal_Metabolic
Inflammatory_Hematological
Hepatic_Coagulation
```

This reduced the number of clinical predictors while retaining information from multiple physiological and laboratory measurements.

### 5. Temporal variables contributed strongly

Variables such as `ICULOS` and `Hour` showed substantial model importance.

This indicates that the temporal context of ICU observations plays an important role in the learned prediction patterns.

However, these variables may also capture ICU stay duration, measurement patterns, or dataset structure rather than purely physiological changes.

### 6. Threshold selection affects performance

Changing the classification threshold produced different precision-recall trade-offs.

This is particularly important in highly imbalanced medical prediction problems.

---

# ⚠️ Limitations

This project has several limitations.

* The model was developed using a single dataset.
* Results may not generalize to other hospitals or patient populations.
* The dataset contains substantial missing clinical measurements.
* Clinical observations are irregularly sampled.
* The domain reference ranges used for feature engineering are heuristic and were not clinically validated for this project.
* The domain weighting scheme represents project-specific assumptions rather than validated clinical scoring.
* Temporal variables may capture ICU workflow, length of stay, or measurement patterns in addition to physiological information.
* The evaluation was performed retrospectively on an existing dataset.
* No prospective clinical validation was performed.
* Model calibration was not the primary focus of this project.
* The final model should not be interpreted as a clinical diagnostic or decision-support system.

---

# 🚀 Future Improvements

Potential future work includes:

* Hyperparameter optimization
* More advanced temporal feature engineering
* Time-series-specific modeling
* Temporal deep-learning approaches
* Improved missingness modeling
* Probability calibration
* More systematic threshold analysis
* External validation using an independent dataset
* Prospective evaluation
* Additional explainability techniques
* Model monitoring and drift analysis
* Development of a prediction API
* Development of an interactive demonstration interface

---

# 📁 Project Notebooks

The project is organized into six main notebooks.

### `01_eda.ipynb`

Exploratory analysis of:

* Dataset structure
* Missing values
* Class imbalance
* Patient-level statistics
* Outliers
* Correlations
* Feature distributions
* Temporal patterns

### `02_data_preprocessing.ipynb`

Includes:

* Patient-level train/validation/test splitting
* Patient-wise forward filling
* Clinical-domain feature engineering
* Domain-level deviation scores
* Feature selection
* Final processed dataset generation

### `03_modeling_xgboost.ipynb`

Includes:

* XGBoost model development
* Class imbalance handling
* Validation analysis
* Threshold analysis
* Feature importance
* Permutation importance
* SHAP analysis
* Model saving

### `04_modeling_baselines.ipynb`

Develops and evaluates:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* Linear SVM

### `05_model_comparison.ipynb`

Performs:

* Validation comparison
* ROC-AUC comparison
* PR-AUC comparison
* Threshold analysis
* Final test comparison
* Confusion matrices
* Results export

### `06_final_evaluation.ipynb`

Provides the final focused evaluation of the selected XGBoost model, including:

* Final test predictions
* Classification metrics
* Confusion matrix
* Precision-Recall curve
* ROC curve
* Classification report
* Feature importance
* Final interpretation
* Limitations

---

# 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd sepsis-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

After activating the virtual environment and installing the dependencies, start Jupyter:

```bash
jupyter notebook
```

Then run the notebooks in the following order:

```text
01_eda.ipynb
        ↓
02_data_preprocessing.ipynb
        ↓
03_modeling_xgboost.ipynb
        ↓
04_modeling_baselines.ipynb
        ↓
05_model_comparison.ipynb
        ↓
06_final_evaluation.ipynb
```

The notebooks should be executed sequentially because later notebooks depend on processed datasets, saved models, and evaluation results generated earlier in the workflow.

---

# 📄 Results and Summaries

Detailed project findings are documented in the `summary/` directory:

```text
summary/
├── 01_eda_summary.md
├── 02_preprocessing_summary.md
├── 03_xgboost_modeling_summary.md
├── 04_baseline_models_summary.md
├── 05_model_comparison_summary.md
└── 06_final_evaluation_summary.md
```

Model comparison results are stored in:

```text
results/
├── validation_model_comparison.csv
├── test_model_comparison.csv
└── xgboost_classification_report.txt
```

---

# 👨‍💻 Author

**Dipesh Ghimire**

Data Science & Machine Learning Intern

* **GitHub:** [Add GitHub profile]
* **LinkedIn:** [Add LinkedIn profile]

---

# 📄 License

This project was developed for **educational and internship purposes**.

The dataset remains subject to the terms, conditions, and licensing requirements of its original source.

---

## ⚠️ Disclaimer

This project is a machine learning research and educational project.

The predictions generated by the model **must not be interpreted as medical diagnoses, clinical recommendations, or treatment decisions**. Real-world clinical use would require appropriate clinical validation, regulatory review, prospective testing, and integration with qualified healthcare professionals.
