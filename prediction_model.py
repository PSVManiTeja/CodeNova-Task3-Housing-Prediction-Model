"""
================================================================================
 CODENOVA AI/ML INTERNSHIP - TASK 3: PREDICTION MODEL (REGRESSION & ML)
 California Housing Price Prediction with Rigorous Feature Engineering
 Author: CodeNova AI/ML Intern
 Tech Stack: Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# 1. SETUP & DIRECTORIES
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("     CODENOVA AI/ML INTERNSHIP - BATCH 01: TASK 3 (PREDICTION MODEL)")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "california_housing.csv")

# ------------------------------------------------------------------------------
# 2. DATA INGESTION & EXPLORATORY SUMMARY
# ------------------------------------------------------------------------------
if not os.path.exists(CSV_FILE):
    print("[*] Local dataset not detected. Downloading California Housing dataset...")
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df.to_csv(CSV_FILE, index=False)
    print(f"    -> Cached to: {CSV_FILE}")
else:
    df = pd.read_csv(CSV_FILE)
    print(f"[*] Ingested existing local dataset: {CSV_FILE}")

print(f"[*] Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(f"[*] Missing Values Detected: {null_counts[null_counts > 0].to_dict()} (Handled via Median Imputation)")

# ------------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (Domain-Specific Engineered Attributes)
# ------------------------------------------------------------------------------
print("\n[*] Performing Feature Engineering...")
# We add intuitive physical ratios that correlate strongly with house valuation
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

print("    -> Engineered 'rooms_per_household' (total_rooms / households)")
print("    -> Engineered 'bedrooms_per_room'   (total_bedrooms / total_rooms)")
print("    -> Engineered 'population_per_household' (population / households)")

# Target and Features
TARGET_COL = 'median_house_value'
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ------------------------------------------------------------------------------
# 4. LEAKAGE-FREE PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
# Identify categorical vs numerical columns
cat_cols = ['ocean_proximity']
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"[*] Numerical features ({len(num_cols)}): {num_cols}")
print(f"[*] Categorical features ({len(cat_cols)}): {cat_cols}")

# Numerical transformer: Median Imputation + StandardScaler
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: OneHotEncoder
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full preprocessor ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# Strict 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"[*] Train partition: {len(X_train)} samples | Test partition: {len(X_test)} samples")

# ------------------------------------------------------------------------------
# 5. MODEL TRAINING & COMPARATIVE BENCHMARKING
# ------------------------------------------------------------------------------
print("\n[*] Training and benchmarking baseline vs. ensemble estimators...")

candidate_models = {
    "Baseline (Dummy Mean)": DummyRegressor(strategy='mean'),
    "Ridge Regression (Linear Baseline)": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
    "HistGradientBoosting Regressor": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=31, random_state=42)
}

benchmark_results = []
trained_pipelines = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in candidate_models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # 5-Fold Cross-Validation on training data (negative RMSE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse = -cv_scores.mean()

    # Fit on complete training data
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Evaluate on held-out test data
    y_pred = pipe.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    benchmark_results.append({
        "Model": name,
        "CV RMSE ($)": cv_rmse,
        "Test RMSE ($)": test_rmse,
        "Test MAE ($)": test_mae,
        "R2 Score": test_r2,
        "y_pred": y_pred
    })

# Format and display benchmark table
print("\n" + "=" * 80)
print("            OFFICIAL PREDICTION BENCHMARK TABLE (HELD-OUT TEST DATA)")
print("=" * 80)
print(f"{'Model':<35} | {'Test RMSE ($)':<13} | {'Test MAE ($)':<13} | {'R2 Score':<9} | {'Performance Gain'}")
print("-" * 80)
baseline_rmse = benchmark_results[0]["Test RMSE ($)"]
for b in benchmark_results:
    gain = f"+{((baseline_rmse - b['Test RMSE ($)']) / baseline_rmse) * 100:.1f}% vs Mean" if b['Test RMSE ($)'] < baseline_rmse else "Baseline"
    print(f"{b['Model']:<35} | ${b['Test RMSE ($)']:>11,.0f} | ${b['Test MAE ($)']:>11,.0f} | {b['R2 Score']:>8.4f}  | {gain}")
print("=" * 80)

# Select best model
best_name = "HistGradientBoosting Regressor" if benchmark_results[3]["R2 Score"] > benchmark_results[2]["R2 Score"] else "Random Forest Regressor"
best_pipe = trained_pipelines[best_name]
print(f"\n[*] Best Performing Architecture: {best_name}")

# Serialize best model
joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_housing_pipeline.joblib"))
print(f"[*] Serialized best pipeline to: models/best_housing_pipeline.joblib")

# ------------------------------------------------------------------------------
# 6. EVALUATION CHARTS & FEATURE IMPORTANCES
# ------------------------------------------------------------------------------
print("\n[*] Generating high-resolution regression diagnostic plots...")
best_res = next(r for r in benchmark_results if r["Model"] == best_name)
y_pred_best = best_res["y_pred"]
residuals = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test / 1000, y_pred_best / 1000, alpha=0.25, color='#2b5c8f', edgecolors='none', s=25)
axes[0].plot([0, 500], [0, 500], 'r--', lw=2, label='Ideal 1:1 Parity')
axes[0].set_title(f'Actual vs. Predicted Values\n({best_name})', fontweight='bold', fontsize=11)
axes[0].set_xlabel('Actual Median House Value ($1,000s)', fontsize=10)
axes[0].set_ylabel('Predicted Median House Value ($1,000s)', fontsize=10)
axes[0].legend(loc='upper left')
axes[0].set_xlim(0, 520)
axes[0].set_ylim(0, 520)

# Plot 2: Residuals Distribution
sns.histplot(residuals / 1000, bins=50, kde=True, color='#2ca02c', ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
axes[1].set_title('Residuals Error Distribution (Actual - Predicted)', fontweight='bold', fontsize=11)
axes[1].set_xlabel('Prediction Error ($1,000s)', fontsize=10)
axes[1].set_ylabel('Frequency', fontsize=10)

# Plot 3: Model Comparison Bar Chart (RMSE)
names_short = ["Mean Baseline", "Ridge Linear", "Random Forest", "Gradient Boost"]
rmses = [r["Test RMSE ($)"] for r in benchmark_results]
bar_colors = ['#7f7f7f', '#aec7e8', '#1f77b4', '#2ca02c']
bars = axes[2].bar(names_short, rmses, color=bar_colors)
axes[2].set_title('Root Mean Squared Error (RMSE) Comparison\n(Lower is Better)', fontweight='bold', fontsize=11)
axes[2].set_ylabel('RMSE in Dollars ($)', fontsize=10)
axes[2].set_ylim(0, max(rmses) * 1.15)
for bar in bars:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 1500, f"${int(yval):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig_eval_path = os.path.join(PLOTS_DIR, "prediction_model_evaluation.png")
plt.savefig(fig_eval_path, dpi=300)
plt.close()
print(f"    -> Exported: {fig_eval_path}")

# Plot Feature Importances from Random Forest
rf_estimator = trained_pipelines["Random Forest Regressor"].named_steps['model']
preprocessor_fitted = trained_pipelines["Random Forest Regressor"].named_steps['preprocessor']

cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
all_features = num_cols + cat_feature_names
importances = rf_estimator.feature_importances_

indices = np.argsort(importances)[::-1]
top_n = min(12, len(all_features))

plt.figure(figsize=(10, 5.5))
sns.barplot(x=importances[indices[:top_n]], y=[all_features[i] for i in indices[:top_n]], palette='Blues_r')
plt.title('Top Feature Importances (Random Forest Regressor)', fontweight='bold', fontsize=12)
plt.xlabel('Gini Importance Weight', fontsize=10)
plt.ylabel('Feature', fontsize=10)
plt.tight_layout()
fig_fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(fig_fi_path, dpi=300)
plt.close()
print(f"    -> Exported: {fig_fi_path}")

# ------------------------------------------------------------------------------
# 7. REAL-TIME SAMPLE PREDICTION DEMONSTRATION & LIMITATIONS ANALYSIS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("          SAMPLE PREDICTION DEMONSTRATION ON HELD-OUT TEST PROPERTIES")
print("=" * 80)
print(f"{'INDEX':<6} | {'ACTUAL PRICE':<14} | {'PREDICTED PRICE':<16} | {'ERROR ($)':<12} | {'ACCURACY %':<10}")
print("-" * 80)
sample_indices = X_test.index[:6]
for idx in sample_indices:
    actual = y_test.loc[idx]
    sample_df = X_test.loc[[idx]]
    pred = float(best_pipe.predict(sample_df)[0])
    err = pred - actual
    pct_err = abs(err) / actual * 100
    acc_pct = max(0.0, 100.0 - pct_err)
    print(f"#{idx:<5} | ${actual:>12,.0f} | ${pred:>14,.0f} | ${err:>+10,.0f} | {acc_pct:>8.1f}%")
print("=" * 80)

print("\n" + "=" * 80)
print("            METHODOLOGY, MODEL LIMITATIONS & ASSUMPTIONS")
print("=" * 80)
print("1. Target Price Capping at $500,000:")
print("   The original California census dataset capped median_house_value at $500,001.")
print("   This produces horizontal clipping in scatter residuals above $500k.")
print("2. Macroeconomic Omissions:")
print("   The model relies on census tract geographic/structural attributes but lacks")
print("   micro-neighborhood indicators (school ratings, mortgage interest rates, crime rate).")
print("3. Zero Data Leakage Compliance:")
print("   All imputation, standard scaling, and one-hot encoding are strictly encapsulated")
print("   inside scikit-learn Pipeline objects and fitted solely on the training partition.")
print("=" * 80)

print("\n[TIP] Both Tasks (Spam Classifier & Prediction Model) are fully completed!")
print("      Execute: python spam_classifier.py")
print("      Execute: python prediction_model.py")
print("All tasks executed successfully! Take screenshots of this terminal execution.")
