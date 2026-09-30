# Model Comparison Summary

## 1. Objective

The purpose of this notebook was to compare the machine learning models developed for sepsis prediction using the same patient-level train/validation/test split and the final 10-feature representation developed during preprocessing.

The following models were evaluated:

* XGBoost
* Random Forest
* HistGradientBoosting
* Logistic Regression
* Linear SVM

Because the dataset is highly imbalanced, with sepsis observations representing a small proportion of the total observations, **Precision-Recall AUC (PR-AUC)** was used as the primary model comparison metric.

The validation set was used for model comparison and classification threshold selection. The test set was then used for final performance evaluation using the selected thresholds.

---

## 2. Final Feature Set

All models used the same final feature set:

### Clinical domain features

* Hemodynamic
* Respiratory
* Renal_Metabolic
* Inflammatory_Hematological
* Hepatic_Coagulation

### Temporal features

* Hour
* ICULOS
* HospAdmTime

### Categorical features

* Gender
* ICU_Unit

This reduced the original clinical feature space to a compact 10-feature representation.

`Patient_ID` was retained in the processed datasets for patient-level traceability but was not used as a model input.

---

## 3. Preprocessing Used for Model Comparison

The core preprocessing preserved missing values in the clinical domain features.

Different models required different missing-value handling:

* **XGBoost:** native missing-value handling.
* **HistGradientBoosting:** native missing-value handling.
* **Random Forest:** median imputation fitted only on the training data.
* **Logistic Regression:** median imputation fitted only on the training data.
* **Linear SVM:** median imputation fitted only on the training data.

No StandardScaler was used for the baseline models.

The median imputer was fitted using training data only and then applied to the validation and test sets to avoid data leakage.

---

## 4. Validation Results

The validation set was used to compare model performance and select classification thresholds.

| Model                |   PR-AUC |  ROC-AUC | Threshold | Precision |   Recall |       F1 |
| -------------------- | -------: | -------: | --------: | --------: | -------: | -------: |
| XGBoost              | 0.091780 | 0.785275 |      0.80 |  0.141187 | 0.278458 | 0.187371 |
| Random Forest        | 0.085073 | 0.781099 |      0.80 |  0.141078 | 0.285663 | 0.188877 |
| HistGradientBoosting | 0.083325 | 0.782045 |      0.80 |  0.133402 | 0.281700 | 0.181060 |
| Logistic Regression  | 0.073398 | 0.735228 |      0.75 |  0.097195 | 0.265850 | 0.142347 |
| Linear SVM           | 0.070716 | 0.735272 |      0.00 |  0.043701 | 0.592579 | 0.081399 |

### Validation observations

XGBoost achieved the highest validation PR-AUC at **0.091780**.

Random Forest achieved a very similar F1-score to XGBoost at the selected threshold. HistGradientBoosting also showed comparable performance.

Logistic Regression and Linear SVM produced lower PR-AUC values than the tree-based models.

Linear SVM produced substantially higher recall at its natural decision threshold, but this came with substantially lower precision.

---

## 5. Threshold Selection

Classification thresholds were selected using the validation set rather than the test set.

The selected thresholds were:

| Model                | Selected Threshold |
| -------------------- | -----------------: |
| XGBoost              |               0.80 |
| Random Forest        |               0.80 |
| HistGradientBoosting |               0.80 |
| Logistic Regression  |               0.75 |
| Linear SVM           |               0.00 |

For the probability-based models, thresholds were evaluated over a grid from 0.10 to 0.90.

The threshold producing the highest tested validation F1-score was selected for each probability-based model.

Linear SVM uses a decision function rather than class probabilities, so its natural decision threshold of 0 was used.

The threshold analysis demonstrates that changing the classification threshold produces a trade-off between precision and recall, which is particularly important for highly imbalanced classification problems.

---

## 6. Final Test Results

The selected validation thresholds were applied to the held-out test set.

| Model                |   PR-AUC |  ROC-AUC | Threshold | Precision |   Recall |       F1 | Accuracy |
| -------------------- | -------: | -------: | --------: | --------: | -------: | -------: | -------: |
| XGBoost              | 0.092856 | 0.791266 |      0.80 |  0.135839 | 0.248638 | 0.175692 | 0.955805 |
| Random Forest        | 0.091238 | 0.787584 |      0.80 |  0.139223 | 0.256301 | 0.180434 | 0.955896 |
| HistGradientBoosting | 0.088892 | 0.786422 |      0.80 |  0.128011 | 0.245232 | 0.168214 | 0.954060 |
| Logistic Regression  | 0.073414 | 0.743265 |      0.75 |  0.106081 | 0.240634 | 0.147249 | 0.947205 |
| Linear SVM           | 0.071523 | 0.744026 |      0.00 |  0.050508 | 0.609843 | 0.093290 | 0.775450 |

### Test observations

The test results were broadly consistent with the validation results.

XGBoost achieved a test PR-AUC of **0.092856** and ROC-AUC of **0.791266**.

Random Forest achieved a test PR-AUC of **0.091238**, with precision of **0.139223**, recall of **0.256301**, and F1-score of **0.180434** at the selected threshold.

HistGradientBoosting produced a test PR-AUC of **0.088892**.

Logistic Regression produced a test PR-AUC of **0.073414**.

Linear SVM produced the highest recall among the evaluated models at its selected threshold, but also produced substantially more false positives, resulting in a lower F1-score.

Accuracy was high for most models because the dataset is highly imbalanced. Therefore, accuracy was not treated as the primary metric.

---

## 7. Confusion Matrices

The final test confusion matrices were:

### XGBoost

```text
[[294833   9288]
 [  4412   1460]]
```

* True Negatives: 294,833
* False Positives: 9,288
* False Negatives: 4,412
* True Positives: 1,460

### Random Forest

```text
[[294816   9305]
 [  4367   1505]]
```

* True Negatives: 294,816
* False Positives: 9,305
* False Negatives: 4,367
* True Positives: 1,505

### HistGradientBoosting

```text
[[294312   9809]
 [  4432   1440]]
```

* True Negatives: 294,312
* False Positives: 9,809
* False Negatives: 4,432
* True Positives: 1,440

### Logistic Regression

```text
[[292214  11907]
 [  4459   1413]]
```

* True Negatives: 292,214
* False Positives: 11,907
* False Negatives: 4,459
* True Positives: 1,413

### Linear SVM

```text
[[236803  67318]
 [  2291   3581]]
```

* True Negatives: 236,803
* False Positives: 67,318
* False Negatives: 2,291
* True Positives: 3,581

The confusion matrices illustrate the precision-recall trade-off observed during threshold analysis. In particular, Linear SVM identifies substantially more positive observations but also generates substantially more false positives.

---

## 8. Model Comparison Interpretation

The tree-based models showed relatively similar performance on the validation and test sets.

XGBoost produced the highest PR-AUC among the evaluated models on both validation and test data. Random Forest produced a very similar test PR-AUC and a slightly higher F1-score at its selected threshold.

HistGradientBoosting also performed competitively but produced slightly lower PR-AUC and F1 values than XGBoost and Random Forest.

The linear models produced lower PR-AUC values, indicating weaker ranking performance for this dataset and feature representation.

The results also show that classification threshold has a substantial effect on precision and recall. Therefore, the default probability threshold of 0.50 should not automatically be assumed to be appropriate for an imbalanced prediction problem.

---

## 9. Important Evaluation Considerations

The patient-level split was used to reduce the risk of having observations from the same patient appear in different dataset partitions.

This is important because each patient contributes multiple sequential observations. A random row-level split could otherwise allow information from the same patient to appear in both training and validation/test sets.

The validation set was used for:

* Model comparison
* Threshold analysis
* Threshold selection

The test set was used for:

* Final performance reporting
* Confusion-matrix analysis

No threshold tuning was performed using the test results.

One limitation is that the XGBoost test set had already been evaluated earlier during the XGBoost modeling notebook. Therefore, the test set should not be described as having remained completely unseen throughout the entire project. The test results are reported as final evaluation results, while validation performance remains the basis for model comparison and threshold selection.

---

## 10. Limitations

Several limitations should be considered when interpreting these results:

1. **Severe class imbalance**

   Sepsis observations represent only a small proportion of all observations. This makes accuracy less informative than PR-AUC, precision, recall, and F1-score.

2. **High missingness**

   Many clinical laboratory variables contain substantial missingness. The preprocessing strategy preserved missingness rather than aggressively imputing all clinical measurements.

3. **Temporal dependence**

   Multiple observations belong to the same patient and occur sequentially over time. Although patient-level splitting reduces leakage between partitions, observations within each patient remain temporally dependent.

4. **Feature engineering assumptions**

   The clinical domain features and reference-range deviations were constructed for this educational/research project. They should not be interpreted as validated clinical scoring systems.

5. **Threshold dependence**

   Precision and recall depend strongly on the classification threshold. Different applications may require different operating points.

6. **External validation**

   The models were evaluated on the available dataset split only. External validation on an independent clinical dataset would be necessary before making claims about generalization to other populations or healthcare settings.

7. **Clinical interpretation**

   The models are intended for educational and research purposes. They are not clinical diagnostic systems and should not be used to make medical decisions.

---

## 11. Overall Project Outcome

The project developed a complete machine-learning workflow for sepsis prediction:

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Patient-Level Data Splitting
     ↓
Patient-Wise Forward Filling
     ↓
Clinical Domain Feature Engineering
     ↓
Dimensionality Reduction
     ↓
Final 10-Feature Dataset
     ↓
XGBoost Modeling
     ↓
Baseline Model Development
     ↓
Validation-Based Model Comparison
     ↓
Threshold Selection
     ↓
Final Test Evaluation
```

The comparison demonstrated that the tree-based models captured useful nonlinear relationships in the engineered clinical and temporal features, while the linear models provided useful baseline references with different precision-recall behavior.

The final evaluation provides a basis for discussing model performance, threshold trade-offs, feature engineering, class imbalance, and the limitations of machine-learning approaches for this type of medical prediction task.

All results are intended for educational and research purposes and should not be interpreted as clinical diagnostic performance.
