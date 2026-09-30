# Modeling Summary

## 1. Modeling Objective

The modeling stage aimed to develop a machine-learning model capable of predicting sepsis observations using the processed clinical, temporal, demographic, and ICU-related features.

The primary modeling objectives were to:

* Train a baseline classification model
* Account for severe class imbalance
* Evaluate predictive performance using appropriate classification metrics
* Investigate alternative model configurations
* Select the strongest configuration using validation data
* Evaluate the final model on an unseen test set
* Investigate model behaviour using feature importance, permutation importance, and SHAP

---

## 2. Model Input

The final model used the following 10 features:

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

`Patient_ID` was excluded from the model because it is an identifier rather than a predictive clinical feature.

The target variable was:

* `SepsisLabel`

---

## 3. Class Imbalance

The training data contained a severe class imbalance:

| Class      |     Count | Percentage |
| ---------- | --------: | ---------: |
| Non-Sepsis | 1,067,723 |     98.23% |
| Sepsis     |    19,268 |      1.77% |

The ratio between negative and positive observations was approximately **55.41:1**.

To account for this imbalance, the XGBoost model used the `scale_pos_weight` parameter:

`scale_pos_weight = 55.41`

This value was calculated using the training data only.

---

## 4. Baseline XGBoost Model

XGBoost was selected as the primary modeling algorithm because it:

* Handles nonlinear relationships
* Can model feature interactions
* Supports missing numerical values natively
* Performs well on tabular data
* Allows class weighting
* Provides multiple model interpretation methods

The baseline configuration was:

| Parameter        |           Value |
| ---------------- | --------------: |
| Model            |   XGBClassifier |
| n_estimators     |             300 |
| max_depth        |               6 |
| learning_rate    |            0.05 |
| subsample        |             0.8 |
| colsample_bytree |             0.8 |
| scale_pos_weight |           55.41 |
| objective        | binary:logistic |
| eval_metric      |           aucpr |
| tree_method      |            hist |
| random_state     |              42 |

PR-AUC was used as the primary evaluation metric during training because of the severe class imbalance.

---

## 5. Baseline Learning Behaviour

During training, training PR-AUC continued to increase while validation PR-AUC peaked early and subsequently declined.

The best validation performance occurred around iteration **47**.

The best validation PR-AUC from the learning curve was approximately:

**0.09136**

This behaviour indicated that the model began to overfit after the early portion of training.

The model was therefore evaluated using the best validation iteration when appropriate during validation analysis.

---

## 6. Validation Performance

Using the selected validation threshold of **0.80**, the model achieved approximately:

| Metric    | Validation |
| --------- | ---------: |
| PR-AUC    | **0.0918** |
| ROC-AUC   | **0.7853** |
| Precision | **0.0521** |
| Recall    | **0.6261** |
| F1-score  | **0.0962** |
| Accuracy  | **0.7897** |

At the default threshold of 0.50, the model produced relatively high recall but very low precision for the sepsis class.

A threshold analysis was therefore performed on the validation set.

---

## 7. Threshold Analysis

Classification thresholds from **0.10 to 0.90** were investigated.

The highest F1-score among the tested threshold values occurred at approximately:

**Threshold = 0.80**

At this threshold:

* Precision = **0.141**
* Recall = **0.278**
* F1-score = **0.187**

The threshold was selected using validation data rather than the test set.

This threshold was subsequently fixed for the final test evaluation.

---

## 8. Model Development Experiments

Several controlled experiments were performed to determine whether alternative configurations improved validation PR-AUC.

| Experiment                         | Best Validation PR-AUC | Decision     |
| ---------------------------------- | ---------------------: | ------------ |
| **Baseline XGBoost**               |            **0.09178** | **Selected** |
| L2 regularization (`reg_lambda=5`) |                0.09062 | Not selected |
| Maximum depth = 4                  |                0.08828 | Not selected |
| No class weighting                 |                0.08645 | Not selected |
| Clinical-only features             |                0.04508 | Not selected |

### L2 Regularization

Adding `reg_lambda=5` produced a validation PR-AUC of approximately **0.0906**, slightly below the baseline.

Therefore, the additional regularization was not retained.

### Shallower Trees

Reducing `max_depth` from 6 to 4 produced a validation PR-AUC of approximately **0.0883**.

This was lower than the baseline and was therefore not selected.

### No Class Weighting

Removing class weighting reduced validation PR-AUC to approximately **0.0864**.

This demonstrated that accounting for the severe class imbalance improved validation performance.

### Clinical-Only Model

A separate experiment removed the temporal features:

* Hour
* ICULOS
* HospAdmTime

The resulting validation PR-AUC decreased substantially to approximately **0.0451**.

This demonstrated that temporal information contributed substantially to predictive performance in this dataset.

---

## 9. Final Model Selection

The baseline XGBoost configuration achieved the highest validation PR-AUC among the tested configurations.

Therefore, the baseline configuration was retained as the final model.

The final model used:

* 10 model features
* XGBoost
* Class weighting
* Maximum tree depth of 6
* Learning rate of 0.05
* 300 maximum estimators
* Histogram-based tree construction

The test dataset was not used for model selection or threshold tuning.

---

## 10. Final Test Ranking Performance

The final model was evaluated on the previously unseen test patients.

The resulting ranking metrics were:

| Metric      | Test Result |
| ----------- | ----------: |
| **PR-AUC**  |  **0.0881** |
| **ROC-AUC** |  **0.7863** |

Validation and test performance were relatively close:

| Metric  | Validation |   Test |
| ------- | ---------: | -----: |
| PR-AUC  |     0.0918 | 0.0881 |
| ROC-AUC |     0.7853 | 0.7863 |

The relatively similar validation and test values indicate that the model's overall ranking behaviour remained reasonably stable on unseen patients.

---

## 11. Final Test Classification Performance

Using the pre-selected threshold of **0.80**, the final test classification report was:

| Class      | Precision |   Recall | F1-score | Support |
| ---------- | --------: | -------: | -------: | ------: |
| Non-Sepsis |      0.99 |     0.97 |     0.98 | 304,121 |
| Sepsis     |  **0.14** | **0.24** | **0.17** |   5,872 |

Overall:

* Accuracy: **0.96**
* Macro F1-score: **0.58**
* Weighted F1-score: **0.96**

The model performed substantially better for the majority non-sepsis class than for the minority sepsis class.

The sepsis F1-score of **0.17** indicates that minority-class classification remains a major limitation of the current model.

Accuracy should not be interpreted in isolation because approximately 98% of observations belong to the non-sepsis class.

---

## 12. Feature Importance

XGBoost's built-in feature importance produced the following results:

| Feature                    | Importance |
| -------------------------- | ---------: |
| ICULOS                     | **0.3079** |
| Respiratory                | **0.1349** |
| Hour                       | **0.1076** |
| ICU_Unit                   |     0.0800 |
| HospAdmTime                |     0.0779 |
| Renal_Metabolic            |     0.0669 |
| Gender                     |     0.0626 |
| Inflammatory_Hematological |     0.0618 |
| Hepatic_Coagulation        |     0.0536 |
| Hemodynamic                |     0.0468 |

`ICULOS` was the most influential feature according to the model's internal split-based importance.

The three temporal features together accounted for approximately **49.4%** of the built-in feature importance.

This indicates that temporal context was heavily used by the model.

However, feature importance represents model behaviour rather than causal or clinical importance.

---

## 13. Permutation Importance

Permutation importance was calculated on a 20,000-observation sample from the validation set using PR-AUC as the scoring metric.

| Feature                    | Mean Importance | Standard Deviation |
| -------------------------- | --------------: | -----------------: |
| ICULOS                     |      **0.0456** |             0.0043 |
| Hour                       |      **0.0254** |             0.0043 |
| Respiratory                |      **0.0206** |             0.0032 |
| Renal_Metabolic            |          0.0158 |             0.0036 |
| Inflammatory_Hematological |          0.0131 |             0.0032 |
| Hepatic_Coagulation        |          0.0045 |             0.0060 |
| Gender                     |          0.0029 |             0.0016 |
| HospAdmTime                |          0.0015 |             0.0019 |
| Hemodynamic                |          0.0012 |             0.0018 |
| ICU_Unit                   |         -0.0006 |             0.0038 |

`ICULOS` again showed the largest effect on predictive performance.

`Hour` and `Respiratory` also produced substantial decreases in validation PR-AUC when permuted.

In contrast, `ICU_Unit` had an importance value close to zero, indicating that disrupting this feature had little measurable effect on validation PR-AUC.

This demonstrates why multiple feature-importance methods are useful: a feature can have relatively high internal tree importance without producing a large reduction in overall predictive performance when permuted.

---

## 14. SHAP Analysis

SHAP was used to investigate how individual feature values influenced model predictions.

A sample of **5,000 validation observations** was used rather than calculating SHAP values across the entire dataset.

The SHAP beeswarm analysis showed that:

* `ICULOS` had the largest overall contribution.
* Higher `ICULOS` values generally pushed predictions toward higher model output.
* `Respiratory` was among the most influential clinical features.
* Feature effects could occur in both directions depending on the observation.
* Some observations had SHAP values close to zero, indicating relatively small contributions from those features.

The SHAP analysis provides information about the direction and magnitude of feature contributions that cannot be obtained from standard feature importance alone.

SHAP relationships should be interpreted as model associations rather than causal or clinical relationships.

---

## 15. Overall Model Findings

The modeling experiments produced several important findings:

1. The baseline XGBoost configuration performed better than the tested alternatives.
2. Class weighting improved validation PR-AUC.
3. Removing temporal features caused a substantial reduction in validation performance.
4. `ICULOS` was consistently the most influential feature across multiple interpretation methods.
5. `Hour` and `Respiratory` were also consistently important.
6. The model achieved relatively stable ROC-AUC between validation and test datasets.
7. PR-AUC was substantially lower than ROC-AUC, reflecting the difficulty of the highly imbalanced classification problem.
8. The model classified non-sepsis observations much more effectively than sepsis observations.
9. Sepsis precision, recall, and F1-score remained relatively low.
10. The final model therefore demonstrates useful predictive patterns but has significant limitations for minority-class identification.

---

## 16. Model Limitations

Several limitations should be considered when interpreting the results.

### Severe Class Imbalance

Sepsis observations represented only approximately 1.8% of the original observations.

This makes reliable identification of the minority class challenging.

### Temporal Dependence

The model relies substantially on `ICULOS` and `Hour`.

These variables capture temporal patterns in the dataset and may reflect differences in observation timing, patient length of stay, and clinical trajectories.

### Feature Engineering Assumptions

The clinical domain scores were constructed using project-specific reference ranges and weighting assumptions.

They should not be interpreted as validated clinical scoring systems.

### Missing Laboratory Data

Several laboratory measurements remain substantially incomplete even after patient-wise forward filling.

The model relies on XGBoost's native handling of missing values.

### Dataset-Level Generalization

The model was evaluated using held-out patients from the same underlying dataset.

Performance on other hospitals, populations, or clinical settings may differ.

### Clinical Interpretation

The model is an ML research system and should not be interpreted as a validated clinical diagnostic or decision-support system.

---

## 17. Final Modeling Conclusion

The final XGBoost model achieved a test ROC-AUC of **0.7863** and PR-AUC of **0.0881**, indicating that it learned meaningful ranking patterns from the processed clinical time-series data.

However, minority-class classification remained difficult. At the selected threshold of 0.80, the model achieved a sepsis precision of **0.14**, recall of **0.24**, and F1-score of **0.17**.

The model's strongest predictive signals were associated with temporal context, particularly `ICULOS` and `Hour`, together with clinical information from the respiratory, renal/metabolic, and inflammatory/hematological domains.

The modeling stage therefore demonstrates both the potential and limitations of machine-learning-based sepsis prediction using this dataset. The results are suitable for an educational and research-oriented ML project but should not be interpreted as evidence of clinical diagnostic performance.
