# -*- coding: utf-8 -*-
"""
PREDICTIVE ANALYTICS CHALLENGE
Objective: Identify customers most likely to accept an upsell offer. 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==========================================
# 1. DATA LOADING & PORTABILITY
# ==========================================
# We use relative paths. Ensure CSVs are in the same folder as this script.
try:
    train = pd.read_csv("upsell_train.csv")
    # Using sep=";" for the test set as per your observation
    test = pd.read_csv("upsell_test_set_without_TARGET.csv", sep=";")
except FileNotFoundError:
    print("CRITICAL ERROR: Please ensure CSV files are in the script's directory.")

# Standardizing identifier
test = test.rename(columns={"id": "ID"})
test_id = test["ID"].copy()

# ==========================================
# 2. UTILITY FUNCTIONS (Threshold & Ranking)
# ==========================================
def optimize_f1_threshold_unique(y_true, y_proba):
    """Optimizes decision threshold to maximize F1-Score."""
    thresholds = np.unique(y_proba)
    best_f1, best_threshold = -1, 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred)
        if f > best_f1:
            best_f1, best_threshold = f, t
    return best_threshold, best_f1

def capture_rate_top10(y_true, y_proba, top_pct=0.10):
    """Calculates campaign efficiency by ranking the top 10% of scores."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    k = max(1, int(np.floor(top_pct * n)))
    idx = np.argsort(-y_proba)
    top_idx = idx[:k]
    total_positives = y_true.sum()
    if total_positives == 0: return 0.0
    return y_true[top_idx].sum() / total_positives

# ==========================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
y = train["upsell"].astype(int)
x = train.drop(columns=["upsell"])
x_test = test.drop(columns=["ID"])

# Stratified split to maintain class proportions (Upsell=1)
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = ["subscription_type", "region", "device_type"]
num_cols = [c for c in x_tr.columns if c not in cat_cols]

# Imputation using training statistics only (to prevent data leakage)
num_medians = x_tr[num_cols].median()
cat_modes = x_tr[cat_cols].mode().iloc[0]

for df in [x_tr, x_val, x_test]:
    df[num_cols] = df[num_cols].fillna(num_medians)
    df[cat_cols] = df[cat_cols].fillna(cat_modes)

# One-hot encoding and column alignment
Xtr = pd.get_dummies(x_tr, columns=cat_cols).astype(int)
Xval = pd.get_dummies(x_val, columns=cat_cols).reindex(columns=Xtr.columns, fill_value=0).astype(int)
Xtest = pd.get_dummies(x_test, columns=cat_cols).reindex(columns=Xtr.columns, fill_value=0).astype(int)

# Scaling (Mandatory for KNN and SVM to calculate distances correctly)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xval_s = scaler.transform(Xval)
Xtest_s = scaler.transform(Xtest)

# ==========================================
# 4. MODELING & METRIC EXTRACTION
# ==========================================
results = {}

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr.fit(Xtr, y_tr)
results['LOGIT'] = (lr.predict_proba(Xval)[:, 1], lr.predict_proba(Xtest)[:, 1], False)

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=8, class_weight="balanced", random_state=42)
rf.fit(Xtr, y_tr)
results['RF'] = (rf.predict_proba(Xval)[:, 1], rf.predict_proba(Xtest)[:, 1], False)

# Model 3: XGBoost (Maximized for Bonus)
ratio = float(np.sum(y_tr == 0)) / np.sum(y_tr == 1)
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=10, scale_pos_weight=ratio, random_state=42, eval_metric='logloss')
xgb.fit(Xtr, y_tr)
results['XGBOOST'] = (xgb.predict_proba(Xval)[:, 1], xgb.predict_proba(Xtest)[:, 1], False)

# Model 4: Gradient Boosting (Replaces KNN for better probability granularity)
hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=5, random_state=42)
hgb.fit(Xtr, y_tr)
results['GRADIENT_BOOSTING'] = (hgb.predict_proba(Xval)[:, 1], hgb.predict_proba(Xtest)[:, 1], False)

# Model 5: SVM (On Scaled Data with Probability=True)
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm.fit(Xtr_s, y_tr)
results['SVM'] = (svm.predict_proba(Xval_s)[:, 1], svm.predict_proba(Xtest_s)[:, 1], True)

# ==========================================
# 5. FINAL EVALUATION & EXCEL GENERATION
# ==========================================
output_filename = "Group_09_Upsell_Predictions.xlsx"

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    for model_name, (val_proba, test_proba, _) in results.items():
        # Optimize threshold on validation data
        t, f1 = optimize_f1_threshold_unique(y_val, val_proba)
        cap10 = capture_rate_top10(y_val, val_proba)
        
        # Displaying performance breakdown per model
        val_preds = (val_proba >= t).astype(int)
        print(f"\n--- {model_name} PERFORMANCE ---")
        print(f"Threshold: {t:.4f} | F1: {f1:.4f} | Capture Top 10%: {cap10:.4f}")
        print(classification_report(y_val, val_preds)) # Performance metrics summary
        
        # Apply threshold to test data
        test_class = (test_proba >= t).astype(int)
        
        df_out = pd.DataFrame({
            "ID": test_id.values,
            "Predicted Class": test_class,
            "Predicted Probability": test_proba
        })
        # Strictly following the sheet requirement
        df_out.to_excel(writer, sheet_name=model_name, index=False)


print(f"\nSUCCESS: File '{output_filename}' generated with 5 sheets.")
