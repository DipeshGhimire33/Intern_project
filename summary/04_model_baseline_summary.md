# Baseline Models Summary

## 1. Overview

This stage evaluated several baseline machine learning models for sepsis prediction using the processed dataset prepared during the preprocessing stage.

The purpose of baseline modeling was to compare different model families against the previously developed XGBoost model and determine whether simpler or alternative algorithms could provide competitive predictive performance.

The following models were evaluated:

* Logistic Regression
* Random Forest
* HistGradientBoosting
* Linear Support Vector Machine (Linear SVM)

All models used the same patient-level train/validation/test split and the same final set of 10 model features to ensure a consistent comparison.

The test set was not used for model selection or tuning during this stage.

---

## 2. Dataset and Features

The processed datasets contained:

* Training set: 1,086,991 observations
* Validation set: 155,226 observations
* Test set: 309,993 observations

The final feature set consisted of 10 features:

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

`Patient_ID` was retained in the processed files for patient traceability but was not used as a model feature.

The target variable was `SepsisLabel`.

The target was highly imbalanced, with approximately 98% non-sepsis observations and 2% sepsis observations.

---

## 3. Evaluation Strategy

Because of the severe class imbalance, **Precision-Recall AUC (PR-AUC)** was treated as the primary model comparison metric.

Additional evaluation metrics included:

* ROC-AUC
* Precision
* Recall
* F1-score
* Accuracy

The initial baseline classification results used a probability/decision threshold of 0.5.

Threshold optimization was not performed using the test set. Final threshold selection, if required, will be performed using validation data only.

---

## 4. Missing-Value Handling

The processed dataset intentionally preserved missing clinical domain values because several clinical measurements had substantial missingness.

XGBoost can handle missing values natively, but some baseline models cannot.

For Logistic Regression, Random Forest, and Linear SVM, median imputation was applied.

The imputer was fitted only on the training data:

```python
imputer.fit_transform(X_train)
```

and then applied to validation and test data:

```python
imputer.transform(X_val)
imputer.transform(X_test)
```

This prevented information from the validation or test sets from influencing the imputation values.

Random Forest does not require feature scaling.

Logistic Regression and Linear SVM were standardized after imputation because these models are sensitive to feature scale.

HistGradientBoosting was able to work directly with the preserved missing values and therefore did not require the same imputation step.

---

## 5. Logistic Regression

Logistic Regression was used as a linear baseline.

The model used class balancing to account for the highly imbalanced target:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=200,
    solver="lbfgs",
    n_jobs=-1
)
```

### Validation Results

| Metric    |    Score |
| --------- | -------: |
| PR-AUC    | 0.073398 |
| ROC-AUC   | 0.735228 |
| Precision | 0.042640 |
| Recall    | 0.603746 |
| F1        | 0.079654 |
| Accuracy  | 0.750493 |

The model achieved a recall of approximately 0.60 but had low precision and F1-score.

The PR-AUC of 0.0734 was lower than the tree-based models evaluated later.

---

## 6. Random Forest

Random Forest was evaluated as a nonlinear bagging-based tree model.

Because of the size of the training dataset, a relatively modest configuration was used rather than a large forest.

The model used:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
```

### Validation Results

| Metric    |    Score |
| --------- | -------: |
| PR-AUC    | 0.085073 |
| ROC-AUC   | 0.781099 |
| Precision | 0.058168 |
| Recall    | 0.560519 |
| F1        | 0.105399 |
| Accuracy  | 0.829835 |

Random Forest substantially improved upon Logistic Regression in PR-AUC and ROC-AUC.

It also produced the highest F1-score among the baseline models at the default 0.5 threshold.

---

## 7. HistGradientBoosting

HistGradientBoosting was evaluated as an efficient gradient-boosting model suitable for large tabular datasets.

One advantage of this model is its ability to handle missing values natively.

The model used:

```python
HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)
```

### Validation Results

| Metric    |    Score |
| --------- | -------: |
| PR-AUC    | 0.083325 |
| ROC-AUC   | 0.782045 |
| Precision | 0.053660 |
| Recall    | 0.600504 |
| F1        | 0.098517 |
| Accuracy  | 0.803461 |

HistGradientBoosting produced performance similar to Random Forest in terms of ROC-AUC and PR-AUC.

Its recall was higher than Random Forest, while Random Forest achieved a slightly higher precision and F1-score at the default threshold.

---

## 8. Linear Support Vector Machine

A Linear SVM was evaluated as an alternative linear classification approach.

A kernel-based `SVC` was not used because training a traditional nonlinear SVM on more than one million observations would be computationally expensive.

Instead, `LinearSVC` was used:

```python
LinearSVC(
    class_weight="balanced",
    C=1.0,
    max_iter=1000,
    random_state=42
)
```

Because `LinearSVC` does not provide class probabilities, its `decision_function()` scores were used for PR-AUC and ROC-AUC calculation.

### Validation Results

| Metric    |    Score |
| --------- | -------: |
| PR-AUC    | 0.070716 |
| ROC-AUC   | 0.735272 |
| Precision | 0.043701 |
| Recall    | 0.592579 |
| F1        | 0.081399 |
| Accuracy  | 0.760813 |

Linear SVM produced results similar to Logistic Regression but did not provide an improvement in PR-AUC.

---

## 9. Baseline Model Comparison

The baseline validation results were:

| Model                |   PR-AUC |  ROC-AUC | Precision |   Recall |       F1 | Accuracy |
| -------------------- | -------: | -------: | --------: | -------: | -------: | -------: |
| Logistic Regression  | 0.073398 | 0.735228 |  0.042640 | 0.603746 | 0.079654 | 0.750493 |
| Linear SVM           | 0.070716 | 0.735272 |  0.043701 | 0.592579 | 0.081399 | 0.760813 |
| HistGradientBoosting | 0.083325 | 0.782045 |  0.053660 | 0.600504 | 0.098517 | 0.803461 |
| Random Forest        | 0.085073 | 0.781099 |  0.058168 | 0.560519 | 0.105399 | 0.829835 |
| XGBoost              | 0.091780 | 0.785275 |  0.052131 | 0.626081 | 0.096248 | 0.789732 |

The XGBoost values in this table come from the previously completed XGBoost validation experiment and are included to provide a common reference point.

---

## 10. Observations

Several observations were made from the baseline experiments.

### 10.1 Linear models

Logistic Regression and Linear SVM produced the lowest PR-AUC values.

This indicates that the relationship between the selected features and the sepsis target is not adequately represented by a simple linear decision boundary.

### 10.2 Tree-based models

Random Forest and HistGradientBoosting both performed substantially better than the linear models.

Their higher PR-AUC and ROC-AUC indicate that nonlinear relationships and feature interactions provide useful predictive information.

### 10.3 XGBoost comparison

XGBoost achieved the highest validation PR-AUC among the evaluated models.

Its validation PR-AUC was approximately 0.0918 compared with:

* Random Forest: 0.0851
* HistGradientBoosting: 0.0833
* Logistic Regression: 0.0734
* Linear SVM: 0.0707

The difference between models should be interpreted in the context of the severe class imbalance and the validation-based evaluation strategy.

### 10.4 Classification threshold

The precision, recall, and F1 values shown above were calculated at a threshold of 0.5 for the baseline models.

These threshold-dependent metrics should not be interpreted independently of the chosen threshold.

For the final model comparison, threshold selection will be performed using validation data so that candidate models can be compared under a consistent decision strategy.

---

## 11. Computational Considerations

The dataset contains more than one million training observations, which significantly affects model training time.

For this reason:

* Logistic Regression was used as a computationally simple linear baseline.
* Random Forest was restricted to 50 trees for the initial baseline experiment.
* HistGradientBoosting was selected because it is designed for efficient gradient boosting on tabular data.
* A Linear SVM was used instead of a nonlinear kernel SVM.
* No computationally expensive hyperparameter search was performed for every baseline model.

The goal of this stage was to establish meaningful baseline performance rather than exhaustively optimize every algorithm.

---

## 12. Conclusion

The baseline experiments demonstrated that nonlinear tree-based models performed better than the linear approaches on this dataset.

Logistic Regression and Linear SVM provided useful linear reference models, while Random Forest and HistGradientBoosting provided stronger nonlinear baselines.

XGBoost achieved the highest validation PR-AUC among the evaluated models.

The next stage is to perform a systematic model comparison using the validation results and apply consistent threshold analysis before evaluating the selected candidate model(s) on the untouched test set.

The test set will remain reserved for final evaluation and will not be used for further model selection or threshold tuning.
