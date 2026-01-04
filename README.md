# 🧠 Applied ML – Breast Cancer Classification

An **Applied Machine Learning (Basic)** project on **binary classification** of breast cancer tumors  
using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset and a **Logistic Regression** baseline.

---

## 🎯 Project Summary
- **Task:** Binary classification (benign vs malignant)
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Model:** Logistic Regression (L2 regularization)
- **Preprocessing:** StandardScaler (sklearn Pipeline)
- **Model Selection:** GridSearchCV
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Interpretability:** Feature importance via logistic regression coefficients

---

## 📁 Repository Structure

```
AppliedML-BreastCancer-Classification/
 ├── notebooks/
 │  └──  01_wdbc_logistic_regression.ipynb
 ├── reports/
 │  └──  project_report.pdf
 ├── figures/
 ├── workflow.md
 └── README.md
```

---

## 🚀 Quick Start
1. Install dependencies
   ```
   pip install -r requirements.txt
   ```
   
2. Launch Jupyter
   ```
   jupyter lab
   ```
   
3. Open the notebook
   ```
   notebooks/01_wdbc_logistic_regression.ipynb
   ```

---

## 🧪 Methodology (Overview)

- Stratified Train / Validation / Test split
- Pipeline-based preprocessing to avoid data leakage
- Logistic Regression baseline with L2 regularization
- Hyperparameter tuning using GridSearchCV
- Learning and validation curves for model diagnostics
- Final evaluation on an unseen test set

---

## 📊 Results (Summary)
- High classification performance on unseen test data
- Balanced precision and recall across classes
- Clear and interpretable coefficients highlighting relevant features

Detailed results and discussion are available in the **project report (PDF)**.

---

## 📄 Documentation

- Full step-by-step workflow: workflow.md
- Complete analysis and results: reports/project_report.pdf

---

## 📌 Full Pipeline Documentation

For a complete, step-by-step description of the full machine learning workflow  
(data loading, preprocessing, model training, validation, and evaluation), see:

👉 See [workflow.md](workflow.md)

This file documents the pipeline in detail and complements the high-level overview provided in this README.

---

## 📜 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## 👩‍💻 Author
**Azam Bakhshandeh**  
MSc Bioinformatics — University of Bologna

 
