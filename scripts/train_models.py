"""
Fraud Detection Model Training Script
Trains CatBoost and LightGBM models with proper imbalance handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve, f1_score)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime

print("="*80)
print("FRAUD DETECTION MODEL TRAINING")
print("="*80)

# Load data
print("\n Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
print(f" Loaded {len(df):,} transactions")

# Prepare features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\n Fraud cases: {y.sum():,} ({y.sum()/len(y)*100:.4f}%)")
print(f" Normal cases: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.4f}%)")

# Time-based split (simulate training on past, testing on future)
print("\n  Splitting data chronologically (80% train, 20% test)...")
split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"   Train set: {len(X_train):,} transactions ({y_train.sum()} fraud)")
print(f"   Test set:  {len(X_test):,} transactions ({y_test.sum()} fraud)")

# Calculate class weight
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  Class imbalance ratio: {fraud_ratio:.2f}:1")

# --------------------- TRAIN CATBOOST ---------------------
print("\n" + "="*80)
print(" TRAINING CATBOOST MODEL")
print("="*80)

catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    scale_pos_weight=fraud_ratio,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

print("\n Training CatBoost (this takes 2-3 minutes)...")
catboost_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=100,
    plot=False
)

# Predictions
y_pred_proba_cb = catboost_model.predict_proba(X_test)[:, 1]
y_pred_cb = (y_pred_proba_cb >= 0.5).astype(int)

# Metrics
roc_auc_cb = roc_auc_score(y_test, y_pred_proba_cb)
pr_auc_cb = average_precision_score(y_test, y_pred_proba_cb)
f1_cb = f1_score(y_test, y_pred_cb)

print(f"\n CatBoost Results:")
print(f"   ROC-AUC:  {roc_auc_cb:.4f}")
print(f"   PR-AUC:   {pr_auc_cb:.4f}")
print(f"   F1-Score: {f1_cb:.4f}")

# --------------------- TRAIN LIGHTGBM ---------------------
print("\n" + "="*80)
print(" TRAINING LIGHTGBM MODEL")
print("="*80)

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("\n Training LightGBM (this takes 1-2 minutes)...")
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[
        # Use lightgbm.early_stopping callback properly
    ]
)

# Predictions
y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
y_pred_lgbm = (y_pred_proba_lgbm >= 0.5).astype(int)

# Metrics
roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
pr_auc_lgbm = average_precision_score(y_test, y_pred_proba_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm)

print(f"\n LightGBM Results:")
print(f"   ROC-AUC:  {roc_auc_lgbm:.4f}")
print(f"   PR-AUC:   {pr_auc_lgbm:.4f}")
print(f"   F1-Score: {f1_lgbm:.4f}")

# --------------------- MODEL COMPARISON ---------------------
print("\n" + "="*80)
print(" MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['CatBoost', 'LightGBM'],
    'ROC-AUC': [roc_auc_cb, roc_auc_lgbm],
    'PR-AUC': [pr_auc_cb, pr_auc_lgbm],
    'F1-Score': [f1_cb, f1_lgbm]
})
print(comparison.to_string(index=False))

# Select best model based on PR-AUC (better for imbalanced data)
best_model_name = 'CatBoost' if pr_auc_cb > pr_auc_lgbm else 'LightGBM'
best_model = catboost_model if pr_auc_cb > pr_auc_lgbm else lgbm_model
best_pred_proba = y_pred_proba_cb if pr_auc_cb > pr_auc_lgbm else y_pred_proba_lgbm

print(f"\n Best Model: {best_model_name} (PR-AUC: {max(pr_auc_cb, pr_auc_lgbm):.4f})")

# --------------------- SAVE MODELS ---------------------
print("\n" + "="*80)
print(" SAVING MODELS")
print("="*80)

# Save CatBoost
catboost_model.save_model('models/catboost_fraud_model.cbm')
print(" CatBoost saved to: models/catboost_fraud_model.cbm")

# Save LightGBM
with open('models/lightgbm_fraud_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
print(" LightGBM saved to: models/lightgbm_fraud_model.pkl")

# Save best model designation
with open('models/best_model.txt', 'w') as f:
    f.write(best_model_name)
print(f" Best model designation saved: {best_model_name}")

# Save metrics
metrics = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'catboost': {
        'roc_auc': float(roc_auc_cb),
        'pr_auc': float(pr_auc_cb),
        'f1_score': float(f1_cb)
    },
    'lightgbm': {
        'roc_auc': float(roc_auc_lgbm),
        'pr_auc': float(pr_auc_lgbm),
        'f1_score': float(f1_lgbm)
    },
    'best_model': best_model_name
}

with open('models/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print(" Training metrics saved to: models/training_metrics.json")

# --------------------- VISUALIZATIONS ---------------------
print("\n" + "="*80)
print(" CREATING VISUALIZATIONS")
print("="*80)

# 1. ROC Curves
fpr_cb, tpr_cb, _ = roc_curve(y_test, y_pred_proba_cb)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_pred_proba_lgbm)

plt.figure(figsize=(10, 6))
plt.plot(fpr_cb, tpr_cb, label=f'CatBoost (AUC = {roc_auc_cb:.4f})', linewidth=2)
plt.plot(fpr_lgbm, tpr_lgbm, label=f'LightGBM (AUC = {roc_auc_lgbm:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
print(" ROC curves saved to: figures/roc_curves.png")
plt.close()

# 2. Precision-Recall Curves
precision_cb, recall_cb, _ = precision_recall_curve(y_test, y_pred_proba_cb)
precision_lgbm, recall_lgbm, _ = precision_recall_curve(y_test, y_pred_proba_lgbm)

plt.figure(figsize=(10, 6))
plt.plot(recall_cb, precision_cb, label=f'CatBoost (AP = {pr_auc_cb:.4f})', linewidth=2)
plt.plot(recall_lgbm, precision_lgbm, label=f'LightGBM (AP = {pr_auc_lgbm:.4f})', linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves: Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/precision_recall_curves.png', dpi=300, bbox_inches='tight')
print(" PR curves saved to: figures/precision_recall_curves.png")
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(y_test, (best_pred_proba >= 0.5).astype(int))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title(f'Confusion Matrix: {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(" Confusion matrix saved to: figures/confusion_matrix.png")
plt.close()

# 4. Feature Importance
if best_model_name == 'CatBoost':
    feature_importance = catboost_model.get_feature_importance()
else:
    feature_importance = lgbm_model.feature_importances_

feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(importance_df)), importance_df['importance'].values, color='#3498db')
plt.yticks(range(len(importance_df)), importance_df['feature'].values)
plt.xlabel('Importance Score', fontsize=12)
plt.title(f'Top 15 Feature Importance: {best_model_name}', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
print(" Feature importance saved to: figures/feature_importance.png")
plt.close()

# --------------------- DETAILED REPORT ---------------------
print("\n" + "="*80)
print(" DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, (best_pred_proba >= 0.5).astype(int), 
                          target_names=['Normal', 'Fraud'], digits=4))

print("\n" + "="*80)
print(" TRAINING COMPLETE!")
print("="*80)
print(f"\n Best Model: {best_model_name}")
print(f" Models saved to: models/")
print(f" Figures saved to: figures/")
print(f" Metrics saved to: models/training_metrics.json")
print("\n Next step: Run the FastAPI server!")
print("="*80)