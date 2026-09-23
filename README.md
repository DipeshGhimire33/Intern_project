# 🩺 Sepsis Prediction Using Machine Learning

## 📌 Project Overview

This project was developed as part of my **Data Science and Machine Learning Internship**.

The primary objective of this project is to develop a **machine learning-based system for predicting sepsis from clinical patient data**. The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, and performance evaluation.

Sepsis is a serious medical condition caused by the body's extreme response to an infection. Early identification of patients at risk can potentially support timely clinical assessment and intervention.

> **Note:** This project is intended for educational and research purposes. The model is not a medical diagnostic system and should not be used to make clinical decisions.

---

## 🎯 Objectives

* Understand and analyze the clinical dataset
* Perform data cleaning and preprocessing
* Handle missing and inconsistent values
* Conduct Exploratory Data Analysis (EDA)
* Identify important clinical features associated with sepsis
* Address class imbalance where appropriate
* Build and train machine learning classification models
* Compare the performance of different models
* Evaluate models using clinically relevant classification metrics
* Identify important features contributing to model predictions
* Explore potential improvements for future deployment

---

## 🛠️ Technologies Used

* **Python**
* **NumPy** – Numerical computations
* **Pandas** – Data manipulation and analysis
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine learning and model evaluation
* **Jupyter Notebook** – Development and experimentation

---

## 📂 Project Structure

```text
sepsis-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── sepsis_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── modeling.py
│
├── models/
│   └── trained_models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset used in this project is based on the **PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data**.

**Dataset source:**
Kaggle — Prediction of Sepsis
https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

The dataset contains **clinical measurements collected from patients over time**. Depending on the version of the dataset used, features may include physiological and laboratory measurements such as:

* Heart rate
* Respiratory rate
* Temperature
* Blood pressure
* Oxygen saturation
* White blood cell count
* Lactate
* Platelet count
* Glucose
* Age
* Other clinical measurements

The target variable represents whether the patient meets the dataset's definition of **sepsis**.

---

## 🔍 Methodology

### 1. Data Collection

The dataset was obtained from the Kaggle dataset based on the **PhysioNet/CinC Challenge 2019**.

The raw data was loaded into Python using Pandas and inspected to understand its structure, features, data types, and target variable.

---

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Inspection of missing values
* Handling missing clinical measurements
* Checking for duplicate records
* Identifying inconsistent values
* Handling numerical features
* Feature scaling where required
* Encoding categorical variables, if applicable
* Identifying potential outliers
* Separating features and target variable
* Splitting the data into training and testing datasets

Because clinical datasets can contain substantial missing information, missing-value handling is an important part of the preprocessing pipeline.

---

### 3. Exploratory Data Analysis

EDA was performed to understand the characteristics of the dataset and identify patterns associated with the target variable.

The analysis included:

* Distribution of clinical variables
* Comparison of features between sepsis and non-sepsis cases
* Correlation analysis
* Missing-value analysis
* Outlier analysis
* Target-class distribution
* Feature relationships
* Identification of potentially important predictors

Visualizations were created using **Matplotlib** and **Seaborn**.

---

### 4. Class Imbalance Analysis

Sepsis datasets can contain an imbalance between positive and negative cases.

Therefore, the distribution of the target variable was examined before model training.

Where appropriate, techniques such as:

* Class weighting
* Oversampling
* Undersampling
* SMOTE

can be investigated to improve the model's ability to identify the minority class.

Care must be taken to apply resampling only to the training data to avoid data leakage.

---

### 5. Feature Engineering

Feature engineering was performed to improve the predictive capability of the models.

Possible approaches include:

* Selecting relevant clinical variables
* Creating derived features
* Handling temporal measurements
* Removing highly redundant features
* Scaling numerical variables where required

Feature selection was also considered to reduce unnecessary variables and improve model interpretability.

---

## 🤖 Model Development

Several machine learning classification algorithms were considered for the prediction task.

The models include:

* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **Support Vector Machine (SVM)**

The models were trained using the processed training dataset and evaluated on unseen test data.

---

## 📈 Model Evaluation

Because this is a **medical classification problem**, accuracy alone is not sufficient for evaluating model performance.

The following metrics were considered:

### Accuracy

Measures the overall proportion of correct predictions.

### Precision

Measures how many of the patients predicted as positive were actually positive.

### Recall / Sensitivity

Measures how many of the actual positive cases were correctly identified.

For sepsis prediction, recall is particularly important because missing a positive case can have serious consequences.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between positive and negative cases across different classification thresholds.

### PR-AUC

Precision-Recall AUC can be particularly informative when the positive class is relatively uncommon.

### Confusion Matrix

The confusion matrix was used to examine:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 📊 Results

The trained models were evaluated on the test dataset.

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression |        - |         - |      - |        - |       - |      - |
| Random Forest       |        - |         - |      - |        - |       - |      - |
| XGBoost             |        - |         - |      - |        - |       - |      - |
| SVM                 |        - |         - |      - |        - |       - |      - |

> **Note:** The results should be filled with the actual values obtained from the final test evaluation. No performance values should be reported without running the models.

---

## 💡 Key Insights

The analysis aims to identify clinical variables and patterns associated with sepsis prediction.

Example findings to investigate include:

* Differences in physiological measurements between sepsis and non-sepsis cases
* The impact of missing clinical measurements
* The effect of class imbalance on model performance
* Which features contribute most strongly to model predictions
* The trade-off between precision and recall
* Differences in performance between traditional and ensemble machine learning models

The final insights should be based on the actual results obtained from the dataset.

---

## 🔬 Model Interpretability

For a healthcare-related machine learning project, understanding **why a model makes a prediction** is important.

Feature importance and interpretability techniques can be used to investigate which clinical variables have the greatest influence on model predictions.

Possible approaches include:

* Feature importance from tree-based models
* Permutation importance
* SHAP-based explanations

These methods can help make the model's predictions easier to understand and analyze.

---

## 🚀 Installation

Clone the repository:

```bash
git clone <repository-url>
cd sepsis-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/sepsis_analysis.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model training process.

---

## 📌 Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Advanced feature engineering
* Time-series-based modeling
* Improved handling of missing clinical data
* Investigation of additional machine learning algorithms
* Ensemble modeling
* Model calibration
* Threshold optimization
* Explainable AI techniques such as SHAP
* External validation using an independent dataset
* Development of a prediction API
* Development of an interactive demonstration interface

---

## ⚠️ Limitations

This project has several limitations:

* The model's performance depends on the dataset and preprocessing methodology.
* Clinical datasets may contain missing and irregularly sampled measurements.
* Class imbalance can affect model performance.
* Results obtained from one dataset may not generalize to other hospitals or patient populations.
* A machine learning prediction should not be interpreted as a clinical diagnosis.
* External clinical validation would be required before any real-world medical application.

---

## 👨‍💻 Author

**Dipesh Ghimire**

Data Science & Machine Learning Intern

* **GitHub:** [Add GitHub profile]
* **LinkedIn:** [Add LinkedIn profile]

---

## 📄 License

This project is intended for **educational and internship purposes**.

The dataset remains subject to the terms and conditions of its original source.
