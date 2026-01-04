# 🧭 Workflow — Applied ML Breast Cancer Classification

This document describes the complete **machine learning workflow** used in this project,
from dataset loading to final evaluation on unseen data.

Each step is designed to ensure **reproducibility**, **robust evaluation**, and **clear interpretation**.

---

## 🧪 Step 1: Data loading & sanity checks 

📌 The dataset used in this project is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

It is loaded directly from **scikit-learn**, which guarantees:
- ✅ reproducibility  
- ✅ standardized preprocessing  
- ✅ no manual data corruption  

### 📂 Dataset Source
- `sklearn.datasets.load_breast_cancer`

### 📊 Dataset Characteristics
- 🔢 **Number of samples:** 569
- 🧬 **Number of features:** 30 numerical features  
  (computed from digitized images of fine needle aspirates of breast masses)
- 🎯 **Target variable:**
  - `0` → malignant
  - `1` → benign

### 🧠 Output Objects
After loading, the dataset is split into:
- **X** → feature matrix
- **y** → target label vector

These objects are then passed to the preprocessing and model training pipeline.

---

## 🔀 Step 2: Train / Validation / Test Split

In this step, the dataset is split into **training**, **validation**, and **test** subsets to ensure a reliable and unbiased evaluation of the model.

### 📌 Objectives
- Separate data for **model training**, **hyperparameter tuning**, and **final evaluation**
- Prevent information leakage between training and test phases
- Preserve the original class distribution across all splits

### 🧪 Splitting Strategy
- The dataset is divided into:
  - **Training set**
  - **Validation set**
  - **Test set**
- A **stratified split** is applied to maintain the proportion of **benign** and **malignant** samples in each subset.

### ⚖️ Why Stratification?
Breast cancer datasets often show class imbalance.  
Stratified splitting ensures that:
- Each subset reflects the original class distribution
- Performance metrics are not biased toward the majority class

### 🧠 Role of Each Subset
- **Training set** 🏋️  
  Used to fit the Logistic Regression model.
  
- **Validation set** 🔧  
  Used during model selection and hyperparameter tuning (via cross-validation).
  
- **Test set** 🧪  
  Held out until the very end to provide an unbiased estimate of generalization performance.

### 🚫 Avoiding Data Leakage
- The test set is **never** used during training or model selection.
- All preprocessing steps are later applied through a pipeline to ensure isolation between subsets.

This splitting strategy establishes a solid foundation for reproducible and trustworthy model evaluation.

---

## 🧼 Step 3: Data Preprocessing & Standardization 

In this step, feature preprocessing is applied to prepare the data for effective model training
and to ensure fair contribution of all features.

### 📌 Objectives
- Normalize feature scales to avoid dominance of large-magnitude variables
- Improve numerical stability and optimization of the learning algorithm
- Prevent data leakage during preprocessing

### ⚙️ Preprocessing Strategy
- **Standardization** is applied using **StandardScaler**
- Each feature is transformed to have:
  - zero mean
  - unit variance

This is particularly important for **Logistic Regression**, which is sensitive to feature scale.

### 🔗 Inductive Bias
Standardization ensures that:
- Regularization acts uniformly across features
- Model coefficients are comparable in magnitude
- Optimization converges more reliably

### 🧱 Pipeline-Based Implementation
All preprocessing steps are embedded within an **sklearn Pipeline**, which guarantees that:
- The scaler is fitted **only on training data**
- The same transformation is consistently applied to validation and test sets
- No information from validation or test data leaks into training

### 🚫 Avoiding Data Leakage
- The scaler is **not** fitted independently on validation or test data
- Preprocessing is coupled with the model inside the pipeline

This design ensures a clean and reproducible preprocessing workflow aligned with machine learning best practices.

---

## 🤖 Step 4: Baseline Logistic Regression (with Scaling)

This step corresponds to the **"Baseline Logistic Regression (with scaling)"** stage in the implementation,
where **feature standardization** and **model training** are combined within a single pipeline.

### 🎯 Objective
- Establish a strong and interpretable **baseline classifier**
- Combine preprocessing and model fitting in a clean, reproducible manner
- Provide a reference point for subsequent model tuning

### 🧱 Model Pipeline
The baseline model is implemented using an **sklearn Pipeline** composed of:
- **StandardScaler** → feature standardization
- **LogisticRegression** → linear classification model

This design ensures that all preprocessing steps are learned **only from the training data** and are
consistently applied during validation and testing.

### 🧠 Why Logistic Regression?
Logistic Regression is chosen as a baseline because it:
- Provides a **simple and interpretable** decision boundary
- Handles binary classification efficiently
- Serves as a strong reference model for medical datasets
- Allows direct inspection of feature importance via model coefficients

### ⚙️ Regularization
- **L2 regularization** is applied to control model complexity
- Regularization helps prevent overfitting and improves generalization

At this stage, the model is trained using **default hyperparameters**, before any tuning is applied.

---

## 🔍 Step 5: Hyperparameter Tuning (GridSearchCV)

In this step, hyperparameter tuning is performed to improve the baseline model
and identify the configuration that yields the best generalization performance.

### 🎯 Objective
- Optimize the performance of the Logistic Regression classifier
- Select the most appropriate level of regularization
- Avoid overfitting through cross-validated model selection

### ⚙️ Tuning Strategy
- **GridSearchCV** is used to systematically explore a predefined set of hyperparameters
- The search is performed **only on the training data**
- Cross-validation is applied within the training set to ensure robust selection

### 🔧 Tuned Hyperparameters
The primary hyperparameter tuned is:
- **C** → inverse of regularization strength  
  - Smaller values of `C` increase regularization  
  - Larger values of `C` allow more flexible decision boundaries

This choice directly controls the bias–variance trade-off of the model.

### 🧠 Why GridSearchCV?
GridSearchCV provides:
- An exhaustive and transparent search over hyperparameters
- Cross-validated performance estimates
- Automatic selection of the best-performing configuration

### 🛡️ Preventing Data Leakage
- The validation and test sets are never used during hyperparameter selection
- Preprocessing and model fitting are handled entirely within the pipeline
- This guarantees that tuning decisions are unbiased

At the end of this step, the **best-performing model configuration** is selected
and passed to the final evaluation stage.

---

## 📈 Step 6: Learning Curve & Validation Curve

In this step, diagnostic curves are used to better understand the behavior of the model
and to assess whether it is underfitting, overfitting, or well-balanced.

### 🎯 Objectives
- Evaluate how performance changes as the training set size increases
- Diagnose potential underfitting/overfitting patterns
- Analyze the effect of regularization strength (`C`) on model performance

### 🧪 Learning Curve (Training Size vs Performance)
A **learning curve** shows model performance as a function of the number of training samples.

✅ It helps answer:
- Does the model benefit from more training data?
- Is the gap between training and validation performance large (overfitting)?
- Are both curves low (underfitting)?

Typical interpretation:
- **High train score + low validation score** → potential overfitting
- **Both low** → underfitting
- **Both high and close** → good generalization

### 🔧 Validation Curve (`C` vs Performance)
A **validation curve** evaluates performance as a function of a single hyperparameter.

Here, the curve is computed for **`C`**, the inverse regularization strength:
- Smaller `C` → stronger regularization → simpler model
- Larger `C` → weaker regularization → more flexible model

✅ It helps identify:
- The range of `C` values where validation performance is highest
- Whether extreme regularization (too strong or too weak) hurts performance
- A stable region for selecting the best model

### 🧠 Why These Curves Matter
These diagnostics complement GridSearchCV by:
- Providing intuition about the bias–variance trade-off
- Showing whether more data or different regularization is likely to help
- Supporting model selection decisions with visual evidence

At the end of this step, curve-based diagnostics are used to confirm that the selected configuration is reasonable.

---

## 🧪 Step 7: Final Evaluation on the Test Set

In this step, the selected model is evaluated on the **held-out test set** to obtain
an unbiased estimate of its generalization performance.

### 🎯 Objective
- Assess the final performance of the tuned model on unseen data
- Report standard classification metrics in a fair and transparent manner
- Confirm that model selection decisions generalize beyond the training data

### 🧠 Evaluation Protocol
- The test set is used **only once**, after all training and tuning steps are completed
- No parameters are adjusted based on test performance
- This ensures a realistic estimate of real-world performance

### 📊 Evaluation Metrics
The following metrics are computed:
- **Accuracy** → overall classification performance
- **Precision** → reliability of positive predictions
- **Recall** → sensitivity to malignant cases
- **F1-score** → balance between precision and recall
- **Confusion Matrix** → detailed error analysis

### ⚖️ Why Multiple Metrics?
In medical classification tasks, relying on accuracy alone can be misleading.
Using multiple metrics allows:
- Better understanding of class-specific behavior
- Identification of potential false negatives and false positives
- More informed interpretation of model reliability

### 🧾 Output
This step produces the final performance summary reported in:
- the Jupyter notebook
- the project report (PDF)

These results serve as the basis for the final discussion and conclusions.

---

## 🔍 Step 8: Model Interpretation & Feature Importance

In this final step, the trained Logistic Regression model is interpreted to understand
which features contribute most to the classification decision.

### 🎯 Objective
- Gain insight into the model’s decision-making process
- Identify the most influential features associated with malignant and benign tumors
- Improve transparency and interpretability of the model

### 🧠 Interpretation Strategy
Logistic Regression provides direct access to **model coefficients**, which represent
the contribution of each feature to the prediction.

- Positive coefficients → increase the probability of the **positive class**
- Negative coefficients → decrease the probability of the **positive class**
- The magnitude of a coefficient reflects its relative importance

### 📊 Feature Importance Analysis
- Coefficients are extracted from the trained model within the pipeline
- Features are ranked based on the absolute value of their coefficients
- The most influential features are highlighted and discussed

### ⚖️ Interpretability in Medical Context
Model interpretability is particularly important in medical applications because it:
- Supports trust and transparency
- Helps validate whether the model relies on meaningful biological signals
- Allows domain experts to assess model rememberability and plausibility

### 📌 Final Notes
- Feature importance reflects correlations learned from the data, not causal relationships
- Interpretations should be made cautiously and in conjunction with domain knowledge

This step concludes the workflow by linking predictive performance with interpretability,
providing a complete and transparent machine learning pipeline.
