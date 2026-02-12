# Customer Upsell Predictive Model

This project focuses on identifying customers most likely to accept an upsell offer, optimizing marketing campaign efficiency through data-driven insights.

## Key Methodology
* **Class Imbalance Handling**: Applied `class_weight='balanced'` and `scale_pos_weight` to address the scarcity of positive upsell cases.
* **Model Diversity**: Evaluated 5 different approaches including Logistic Regression, Random Forest, XGBoost, HistGradientBoosting, and SVM.
* **Optimization**: Used a custom function to find the optimal decision threshold that maximizes the **F1-Score**.

## Technical Implementation
* **Language**: Python.
* **Key Libraries**: `scikit-learn` for preprocessing, `XGBoost` for high-performance boosting, and `pandas` for feature engineering.
* **Scalability**: Implemented `StandardScaler` for distance-based models like SVM to ensure correct calculation.

## Business Impact
The model ranks customers by probability, allowing the business to focus on the **Top 10% capture rate**, significantly reducing marketing costs while maximizing conversions.
