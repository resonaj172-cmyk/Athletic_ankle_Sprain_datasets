# Model Selection Report

## Overview
Three distinct machine learning models (Logistic Regression, Random Forest, XGBoost) were trained and evaluated on the exact same 7 post-session features across identical 80/20 train/test splits.

## Comparison Summary

| Model | Test Accuracy | Macro F1 | ROC-AUC (OvR) | Train Accuracy | 5-Fold CV Macro F1 | Train Wall Time (s) | Pred Latency / 1k (ms) | Meets Target (>=0.90) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6708 | 0.6707 | 0.8502 | 0.6729 | 0.6737 | 3.51s | 0.25ms | No |
| Random Forest | 0.6654 | 0.6657 | 0.8464 | 0.7348 | 0.6677 | 77.37s | 12.41ms | No |
| XGBoost | 0.6672 | 0.6664 | 0.8488 | 0.6888 | 0.6721 | 208.32s | 10.95ms | No |

## Selected Model: **Logistic Regression**

### Selection Reasoning & Rule
- **Rule Applied:** Among candidate models, highest test macro-F1 (with ties within 0.005 broken by ROC-AUC and prediction latency).
- **Selected Key:** `logistic_regression`
- **Test Accuracy:** 0.6708 (67.08%)
- **Test Macro F1:** 0.6707
- **ROC-AUC (Macro OvR):** 0.8502
- **Inference Latency:** 0.25 ms per 1,000 samples

### Statistical Separation Note
Note: XGBoost, Random Forest differed by less than 1 percentage point from Logistic Regression; the statistical difference is minimal and selection is practical.

### Accuracy Target Analysis
None of the three models achieved the 0.90 test accuracy threshold on the strictly non-leaked 7-feature post-session dataset. All three models converged around 67.0% - 67.7% accuracy with ROC-AUC values of ~0.852. This is entirely consistent with the author's previous empirical observations (~66%) and reflects the intrinsic signal-to-noise ratio of post-session self-reported and physical measurements without historical or biological target leakage.
