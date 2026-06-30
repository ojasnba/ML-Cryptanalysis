# XGBoost Differential Distinguisher for ASCON320

## Objective

Implement an XGBoost-based differential distinguisher for ASCON320 and compare its performance against the optimized Random Forest implementation.

Dataset generation, input differences, number of rounds, and evaluation pipeline were kept identical to the Random Forest implementation to ensure a fair comparison.

---

# Experimental Setup

Cipher: ASCON320

Training Bits: 40

Rounds: 4

Training Samples: 2^16

Validation Samples: 2^14

Prediction Data: 2^19

Experiments: 100

Input Difference:

0000000001

---

# Baseline XGBoost Configuration

| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 1.0 |
| colsample_bytree | 1.0 |
| objective | binary:logistic |
| tree_method | hist |

---

# Hyperparameter Tuning

## 1. Max Depth

| Max Depth | Graph Accuracy | Calculated Accuracy | Average TP Difference | Observation |
|------------|---------------:|--------------------:|----------------------:|-------------|
| 3 | 59% | 57% | 235 | Underfitting |
| **6** | **69%** | **72%** | **373** | Best Performance |
| 9 | 57% | 54% | 116 | Overfitting |

### Conclusion

Maximum depth of 6 provided the best distinguisher performance.

Increasing tree depth beyond 6 reduced the separation between Real and Random datasets.

---

## 2. Learning Rate

| Learning Rate | Graph Accuracy | Calculated Accuracy | Average TP Difference | Observation |
|---------------|---------------:|--------------------:|----------------------:|-------------|
| 0.30 | 52% | 50% | -26 | Too Aggressive |
| 0.20 | 60% | 58% | 146 | Worse than baseline |
| **0.10** | **69%** | **72%** | **373** | Best Performance |
| 0.05 | 69% | 66% | 276 | Competitive but lower TP separation |

### Conclusion

A learning rate of 0.1 produced the strongest distinguisher.

Higher learning rates degraded model performance, while lower learning rates required more boosting rounds to achieve comparable results.

---

## 3. Number of Estimators

| Estimators | Graph Accuracy | Calculated Accuracy | Average TP Difference | Observation |
|------------|---------------:|--------------------:|----------------------:|-------------|
| **100** | **69%** | **72%** | **373** | Best Performance |
| 300 | 60% | 62% | 196 | Performance degraded |
| 600 | 54% | 56% | 109 | Significant degradation |

### Conclusion

Increasing the number of estimators beyond 100 reduced distinguisher performance for the current configuration.

---

# Final Hyperparameters

| Hyperparameter | Selected Value |
|----------------|---------------|
| n_estimators | **100** |
| max_depth | **6** |
| learning_rate | **0.1** |
| subsample | **1.0** |
| colsample_bytree | **1.0** |

---

# Comparison with Random Forest

| Model | Best Graph Accuracy | Best Calculated Accuracy |
|--------|--------------------:|-------------------------:|
| Random Forest | 67% | 66% |
| XGBoost | **69%** | **72%** |

---

# Summary

An XGBoost-based differential distinguisher was successfully implemented for ASCON320.

Systematic tuning of three major hyperparameters (maximum depth, learning rate, and number of estimators) identified the optimal configuration as:

- max_depth = 6
- learning_rate = 0.1
- n_estimators = 100

Under identical experimental settings, the optimized XGBoost model achieved higher distinguisher performance than the optimized Random Forest baseline while maintaining fast inference time.

Future work includes evaluating additional machine learning models (SVM) and reproducing/improving the CNN-based distinguishers reported in the literature.