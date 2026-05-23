# 🧭 Workflow — Applied ML Breast Cancer Classification

This document describes the complete machine learning workflow used in this project,
from dataset loading to final evaluation and model interpretation.

---

## 🧪 Step 1: Data loading & sanity checks

- Load WDBC dataset from sklearn
- Inspect dataset dimensions and feature names
- Verify class distribution
- Separate features (X) and labels (y)

---

## 📊 Step 1.1: Exploratory Data Analysis (EDA)

- Analyze class imbalance
- Visualize feature distributions
- Plot correlation matrix
- Study separability between benign and malignant samples

---

## 🔀 Step 2: Train / Validation / Test split

- Apply stratified splitting
- Preserve class distribution across subsets
- Create:
  - Training set
  - Validation set
  - Test set

---

## 🤖 Step 3: Baseline Logistic Regression (with scaling)

- Build sklearn Pipeline:
  - StandardScaler
  - LogisticRegression
- Train baseline model
- Evaluate on validation set
- Generate:
  - Confusion matrix
  - ROC curve
  - Precision–Recall curve

---

## 🔍 Step 4: Hyperparameter tuning (GridSearchCV)

- Perform GridSearchCV
- Tune:
  - Regularization parameter (C)
  - Solver
  - Class weights
- Select best model using cross-validation

---

## 📈 Step 5: Learning curve & Validation curve

- Generate learning curves
- Analyze bias / variance behavior
- Evaluate effect of regularization parameter C
- Detect overfitting / underfitting patterns

---

## 🧪 Step 6: Final Evaluation on Test Set

- Evaluate tuned model on unseen test set
- Compute:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - ROC AUC
  - PR AUC

---

## 🔍 Step 7: Feature Importance (Logistic Coefficients)

- Extract logistic regression coefficients
- Rank most important features
- Interpret contribution of features to predictions

---

## 🏁 Step 8: Conclusion

- Summarize model performance
- Discuss generalization ability
- Highlight interpretability and future improvements
