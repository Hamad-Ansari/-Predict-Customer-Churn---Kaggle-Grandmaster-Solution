# 🏆 Predict Customer Churn - Kaggle Grandmaster Solution

**Competition**: <!--citation:1-->  
**Author**: Kaggle Grandmaster Approach  
**Kernel**: [link to your notebook]  
**AUC Score**: **0.915656**

---

## 📌 Project Overview

This repository contains the end-to-end solution for the **Predict Customer Churn** Kaggle competition. The goal is to predict the likelihood of a customer churning based on their telecom service usage and contract details.

We utilized a **Grandmaster strategy** to achieve a **0.9156 AUC** score, incorporating:
*   🧬 **Data Augmentation**: Merging competition data with the original IBM Telco dataset.
*   🛠️ **Feature Engineering**: Financial ratios, service count aggregation, and risk flags.
*   🏗️ **Ensemble Learning**: Combining XGBoost, LightGBM, and CatBoost.
*   ⚡ **GPU Optimization**: Fast training using CUDA/Hist algorithms.

---

## 📊 Performance Summary

| Model               | CV Score | Notes                          |
|---------------------|----------|--------------------------------|
| **XGBoost**         | **0.915656** | GPU-accelerated GBDT          |
| LightGBM            | 0.914969 | Leaf-wise tree growth          |
| CatBoost            | 0.915224 | Ordered boosting               |
| **Best Single Model**| **XGBoost** |                              |
| **Ensemble (1/3 each)**| **0.9157** | Final Submission Score         |

**Training Size**: 601,237 rows (Competition + Original)  
**Feature Count**: 19 engineered features  
**Validation**: 5-Fold Stratified Cross-Validation

---

## 🚀 Key Features & Engineering

### 1. Data Augmentation (`is_generated` flag)
We combined the **594k** rows from the competition dataset with the **7k** rows from the original dataset.
*   Why? Synthetic data rem
   <img width="1790" height="985" alt="3897c5b4-dc0a-44ef-a34b-c6190a501f54" src="https://github.com/user-attachments/assets/15a9ae6e-f26d-49d6-ad35-012beb3f5fb1" />
oves "tail" outliers. The original data restores these hard cases.

### 2. Critical Feature Ratios
```python
# These ratios capture the "financial health" of a customer
df['ChargesPerMonth'] = df['MonthlyCharges']
df['Tenure_Age_Ratio'] = df['tenure'] / (df['SeniorCitizen'] + 1)
df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-5)
