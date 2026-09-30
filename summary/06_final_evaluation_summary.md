# Final Evaluation Summary

## 1. Objective

The purpose of the final evaluation was to assess the performance of the selected XGBoost model on the held-out test set.

Model selection and threshold selection were performed using the validation set. The final evaluation was then performed on the test set using the selected XGBoost configuration.

The final evaluation included:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Confusion matrix
* Precision-Recall curve
* ROC curve
* Classification report
* Feature importance
* Final interpretation and limitations

---

## 2. Selected Model

XGBoost was selected as the final candidate model based on the validation-set model comparison.

The model was evaluated using the first 48 trees, corresponding to the best validation PR-AUC observed during model training.

The classification threshold selected from the validation analysis was:

```text
Threshold = 0.80
```

The final feature set contained 10 features:

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

---

## 3. Final Test-Set Results

The final XGBoost model achieved the following performance on the held-out test set:

| Metric    |   Result |
| --------- | -------: |
| Accuracy  | 0.955805 |
| Precision | 0.135839 |
| Recall    | 0.248638 |
| F1-Score  | 0.175692 |
| ROC-AUC   | 0.791266 |
| PR-AUC    | 0.092856 |

PR-AUC was emphasized because the dataset contains a strong class imbalance.

---

## 4. Confusion Matrix

The final XGBoost confusion matrix was:

```text
[[294833   9288]
 [  4412   1460]]
```

This corresponds to:

|                   | Predicted Non-Sepsis | Predicted Sepsis |
| ----------------- | -------------------: | ---------------: |
| Actual Non-Sepsis |              294,833 |            9,288 |
| Actual Sepsis     |                4,412 |            1,460 |

Therefore:

* True Negatives = 294,833
* False Positives = 9,288
* False Negatives = 4,412
* True Positives = 1,460

The model identified 1,460 of the 5,872 positive observations in the test set at the selected threshold.

---

## 5. Precision-Recall Analysis

The final XGBoost model achieved:

```text
PR-AUC = 0.092856
```

The Precision-Recall curve was used because the sepsis class represents a small proportion of the total observations.

At the selected threshold of 0.80:

```text
Precision = 0.135839
Recall    = 0.248638
F1-Score  = 0.175692
```

This demonstrates the trade-off between precision and recall when detecting the minority sepsis class.

---

## 6. ROC Analysis

The final XGBoost model achieved:

```text
ROC-AUC = 0.791266
```

The ROC curve demonstrates the model's ability to distinguish between sepsis and non-sepsis observations across different classification thresholds.

The ROC-AUC is threshold-independent, while the reported precision, recall, and F1-score depend on the selected threshold of 0.80.

---

## 7. Classification Report

The final classification report showed the following sepsis-class performance:

| Metric    | Sepsis Class |
| --------- | -----------: |
| Precision |     0.135839 |
| Recall    |     0.248638 |
| F1-Score  |     0.175692 |

The results demonstrate that the model identifies a portion of the positive cases while also generating a substantial number of false positive predictions.

---

## 8. Feature Importance

The XGBoost feature importance analysis showed the following approximate ordering:

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

ICULOS was the most influential individual feature in the trained model.

Temporal features, including ICULOS, Hour, and HospAdmTime, also made a substantial contribution to model predictions.

Feature importance represents how the trained model uses the available variables and should not be interpreted as evidence of clinical causality.

---

## 9. Overall Interpretation

The final XGBoost model achieved a test PR-AUC of 0.092856 and ROC-AUC of 0.791266.

The results demonstrate that the model can identify patterns associated with the sepsis label in the test data, but the relatively low precision reflects the difficulty of predicting a rare outcome in a highly imbalanced dataset.

The selected threshold of 0.80 provides a particular precision-recall trade-off and should not be considered a clinically validated decision threshold.

The strong contribution of temporal features also indicates that the timing of observations within the ICU stay is important to the model's predictions.

---

## 10. Limitations

Several limitations should be considered:

1. The dataset is highly imbalanced, with substantially fewer sepsis observations than non-sepsis observations.

2. Many laboratory variables contain substantial missingness.

3. The clinical reference ranges used during feature engineering were heuristic project-level assumptions and were not validated as clinical decision thresholds.

4. The domain scores and feature weighting strategy were designed specifically for this educational research project and are not validated clinical scores.

5. Temporal features such as Hour and ICULOS may capture patterns related to ICU stay duration and data collection processes.

6. The dataset is derived from the PhysioNet/CinC 2019 sepsis prediction challenge and may not represent all hospitals or patient populations.

7. The model has not undergone prospective clinical validation.

8. Test-set performance on this dataset does not establish clinical effectiveness or generalizability to other healthcare environments.

9. The model should not be interpreted as a clinical diagnostic or decision-support system.

---

## 11. Final Conclusion

This project developed an end-to-end machine learning workflow for sepsis prediction using longitudinal ICU data.

The workflow included:

* Exploratory data analysis
* Missing-value analysis
* Patient-level data splitting
* Patient-wise forward filling
* Clinical domain feature engineering
* Dimensionality reduction
* XGBoost modeling
* Baseline model development
* Model comparison
* Threshold selection
* Final test-set evaluation
* Model interpretation

Five models were compared during model development. XGBoost was selected as the final candidate based on validation PR-AUC.

The final XGBoost model achieved:

```text
PR-AUC  = 0.092856
ROC-AUC = 0.791266
F1      = 0.175692
```

on the held-out test set using a classification threshold of 0.80.

Overall, the project demonstrates the challenges of sepsis prediction from highly incomplete, longitudinal ICU data and highlights the importance of patient-level splitting, appropriate evaluation metrics, threshold selection, feature engineering, and careful interpretation of machine learning results.

This project is intended for educational and research purposes and should not be interpreted as a clinical diagnostic system.
