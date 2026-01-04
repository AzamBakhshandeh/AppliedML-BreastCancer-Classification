# Applied ML – Breast Cancer Classification

This repository contains an **Applied Machine Learning (Basic)** project focused on  
**binary classification of breast cancer tumors** using the  
**Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

The project implements a **Logistic Regression baseline** with proper  
data preprocessing, model selection, and evaluation following ML best practices.

---

## Project Overview
- **Task:** Binary classification (benign vs malignant)
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Model:** Logistic Regression (L2 regularization)
- **Framework:** scikit-learn
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Validation:** Stratified Train / Validation / Test split + GridSearchCV

---

## Repository Structure
AppliedML-BreastCancer-Classification/
├── notebooks/
│ └── 01_wdbc_logistic_regression.ipynb
├── reports/
│ └── project_report.pdf
├── figures/
└── README.md

---

## How to Run
1. Install dependencies
 - pip install -r requirements.txt

2. Launch Jupyter
 - jupyter lab

3. Open the notebook
 - notebooks/01_wdbc_logistic_regression.ipynb

---

## Methodology
1. Dataset loading and exploratory analysis
2. Stratified data splitting
3. Feature scaling with StandardScaler
4. Logistic Regression baseline
5. Hyperparameter tuning using GridSearchCV
6. Learning and validation curves
7. Final evaluation on the test set
8. Interpretation of model coefficients

---

## Results (Summary)
- High classification performance on unseen test data
- Balanced precision and recall across classes
- Interpretable model through logistic regression coefficients

Detailed results and discussion are available in the **project report (PDF)**.

---

## Author
**Azam Bakhshandeh**  
MSc Bioinformatics — University of Bologna
