# %% [markdown]
# # V3 Architecture: Hierarchical Macro-Category Classification (SOTA Optimized)
# ----------------------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (V3 Architecture - Mapped to DM Syllabus):
1. DM 3 (Sparsity): Aggregated 24 sparse products into 3 dense clusters.
2. DM 2.5 (Feature Eng): Engineered dynamic temporal features (Lags, Portfolio Growth)
   and granular financial ratios (Age/Tenure, Income/Product).
3. DM 2 (Data Prep): Used Target Encoding for Geography (Province Affinity) and 
   coerced Santander's " NA" string bugs into true nulls.
4. DM 5 (Ensembles): Tuned Random Forest (n=200, depth=20) trained on a natural 
   2.5M row sample. Target shifted from "Ownership" to "Acquisition (Delta)".
5. Kaggle Optimization: Applied post-processing heuristic rules to zero-out 
   probabilities for products the user already owned in T-1.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import time
import gc

print("="*70)
print("V3 PIPELINE: SOTA MACRO-CATEGORY ACQUISITION PREDICTION")
print("="*70)

# [1] Define the Macro-Category Dictionaries
macro_mapping = {
    'Core_Activity':   ['ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_cno_fin_ult1', 'ind_recibo_ult1'],
    'Credit_Spending': ['ind_tjcr_fin_ult1', 'ind_cco_fin_ult1', 'ind_ecue_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1'],
    'Wealth_and_Other':['ind_fond_fin_ult1', 'ind_valo_fin_ult1', 'ind_plan_fin_ult1', 'ind_deco_fin_ult1', 
                        'ind_deme_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctju_fin_ult1', 'ind_cder_fin_ult1', 
                        'ind_ahor_fin_ult1', 'ind_viv_fin_ult1', 'ind_hip_fin_ult1', 'ind_pres_fin_ult1', 
                        'ind_aval_fin_ult1', 'ind_dela_fin_ult1', 'ind_reca_fin_ult1']
}
macro_targets = ['target_core_activity', 'target_credit_spending', 'target_wealth_and_other']

# [2] Load Cleaned Data
print("--- Loading Cleaned Parquet Dataset ---")
df = pd.read_parquet(r"C:\ML2_final_project\train_cleaned.parquet")
df = df.sort_values(by=['ncodpers', 'fecha_dato'])

# [3] Feature Engineering: Context & Demographics
print("--- Engineering Contextual Features & Demographics ---")

# A. Force messy Santander columns to true numeric types
numeric_cols = ['age', 'renta', 'antiguedad', 'ind_actividad_cliente']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# B. Encode Categoricals safely (Explicit 'UNKNOWN' handling)
cat_cols = ['sexo', 'segmento']
le_dict = {}
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('object')
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# C. Basic Contextual Features
if 'renta' in df.columns and 'nomprov' in df.columns:
    prov_median_renta = df.groupby('nomprov')['renta'].transform('median')
    df['renta_relative'] = df['renta'] / prov_median_renta.replace(0, 1)

df['age_to_antiguedad_ratio'] = df['age'] / df['antiguedad'].replace(0, 1)

# [4] Engineer Macro-Targets, Temporal Lags & Acquisition Deltas
print("--- Aggregating Clusters & Engineering Momentum Lags ---")
for macro_name, sub_cols in macro_mapping.items():
    valid_cols = [c for c in sub_cols if c in df.columns]
    df[f'target_{macro_name.lower()}'] = (df[valid_cols].sum(axis=1) > 0).astype(np.int8)

lag_features = []
delta_targets = []

for col in macro_targets:
    # Generate historical lags
    df[f'{col}_lag_1'] = df.groupby('ncodpers')[col].shift(1).fillna(0)
    df[f'{col}_lag_2'] = df.groupby('ncodpers')[col].shift(2).fillna(0)
    df[f'{col}_lag_3'] = df.groupby('ncodpers')[col].shift(3).fillna(0)
    
    # Calculate Momentum & Trend
    df[f'{col}_momentum'] = df[f'{col}_lag_1'] + df[f'{col}_lag_2'] + df[f'{col}_lag_3']
    df[f'{col}_trend'] = df[f'{col}_lag_1'] - df[f'{col}_lag_2']
    
    lag_features.extend([f'{col}_lag_1', f'{col}_lag_2', f'{col}_lag_3', f'{col}_momentum', f'{col}_trend'])

    # CRITICAL LOGIC: The target is ONLY '1' if it is a NEWLY ADDED product
    df[f'{col}_added'] = ((df[col] == 1) & (df[f'{col}_lag_1'] == 0)).astype(np.int8)
    delta_targets.append(f'{col}_added')

# Engineer historical portfolio sizes (Leak-free)
df['total_portfolio_size_lag_1'] = df['target_core_activity_lag_1'] + df['target_credit_spending_lag_1'] + df['target_wealth_and_other_lag_1']
df['total_portfolio_size_lag_2'] = df['target_core_activity_lag_2'] + df['target_credit_spending_lag_2'] + df['target_wealth_and_other_lag_2']
df['portfolio_growth'] = df['total_portfolio_size_lag_1'] - df['total_portfolio_size_lag_2']

# D. Advanced Features (Post-Lag Calculation)
if 'nomprov' in df.columns:
    # Target Encoding: Province Affinity for Wealth Products (Mapped to DM 2.5)
    wealth_prob_by_prov = df.groupby('nomprov')['target_wealth_and_other_lag_1'].mean()
    df['prov_wealth_affinity'] = df['nomprov'].map(wealth_prob_by_prov).fillna(0)

# Under-banked metric
df['income_per_product'] = df['renta'] / df['total_portfolio_size_lag_1'].replace(0, 1)

# Combine Final Feature Space
demographics = ['age', 'renta', 'antiguedad', 'ind_actividad_cliente', 'sexo', 'segmento']
aggregations = ['renta_relative', 'prov_wealth_affinity', 'age_to_antiguedad_ratio', 'income_per_product', 'total_portfolio_size_lag_1', 'portfolio_growth']
base_features = [c for c in (demographics + aggregations + lag_features) if c in df.columns]

X_raw = df[base_features].fillna(0)
y_raw = df[delta_targets] # Training on Acquisition Targets!

del df
gc.collect()

# [5] Train/Test Split & Natural Subsampling
print("--- Splitting and Executing Natural Subsampling (2.5M Rows) ---")
X_train_raw, X_test_final, y_train, y_test_final = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

SAMPLE_SIZE = 2500000
X_train_final = X_train_raw.sample(n=SAMPLE_SIZE, random_state=42)
y_train_final = y_train.loc[X_train_final.index]

del X_train_raw, X_raw, y_raw, y_train
gc.collect()

# [6] Train One-Vs-Rest Random Forest (Optimized Hyperparameters)
print("\n--- Training V3 SOTA Random Forest Ensemble ---")
start_time = time.time()

tuned_rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=15, 
    class_weight='balanced', 
    n_jobs=-1, 
    random_state=42
)
ovr_clf = OneVsRestClassifier(tuned_rf)
ovr_clf.fit(X_train_final, y_train_final)

print(f"Training Execution Time: {time.time() - start_time:.2f} seconds")

# [7] Internal Evaluation
print("\n--- Calculating Internal Validation Metrics ---")
y_pred_binary = ovr_clf.predict(X_test_final)

acc = accuracy_score(y_test_final, y_pred_binary)
prec = precision_score(y_test_final, y_pred_binary, average='macro', zero_division=0)
rec = recall_score(y_test_final, y_pred_binary, average='macro')
f1 = f1_score(y_test_final, y_pred_binary, average='macro')

print("\n" + "="*70)
print("INTERNAL VALIDATION (ACQUISITION INTENT PREDICTION)")
print("="*70)
print(f"Macro Accuracy  : {acc:.4f}")
print(f"Macro Precision : {prec:.4f}")
print(f"Macro Recall    : {rec:.4f}")
print(f"Macro F1-Score  : {f1:.4f}")
print("="*70)

del X_train_final, X_test_final, y_pred_binary
gc.collect()

# ======================================================================
# PHASE 8: EXTERNAL VALIDATION ON MACRO-CATEGORIES (MAY 2016)
# ======================================================================
print("\n" + "="*70)
print("EVALUATING EXTERNAL TEST DATA (OUT-OF-TIME)")
print("="*70)

test_df = pd.read_parquet(r"C:\ML2_final_project\test_cleaned.parquet")

# 1. Clean test set numerics
for col in numeric_cols:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# 2. Apply Categorical Encodings
for col in cat_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('object')
        test_df[col] = test_df[col].fillna('UNKNOWN')
        classes = le_dict[col].classes_
        test_df[col] = test_df[col].astype(str).apply(lambda x: x if x in classes else 'UNKNOWN')
        test_df[col] = le_dict[col].transform(test_df[col])

# 3. Engineer External Set Context & Targets
if 'renta' in test_df.columns and 'nomprov' in test_df.columns:
    prov_median_renta_test = test_df.groupby('nomprov')['renta'].transform('median')
    test_df['renta_relative'] = test_df['renta'] / prov_median_renta_test.replace(0, 1)

test_df['age_to_antiguedad_ratio'] = test_df['age'] / test_df['antiguedad'].replace(0, 1)

if 'nomprov' in test_df.columns:
    # IMPORTANT: Map using the training data probabilities to prevent leakage
    test_df['prov_wealth_affinity'] = test_df['nomprov'].map(wealth_prob_by_prov).fillna(0)

for macro_name, sub_cols in macro_mapping.items():
    valid_cols = [c for c in sub_cols if c in test_df.columns]
    test_df[f'target_{macro_name.lower()}'] = (test_df[valid_cols].sum(axis=1) > 0).astype(np.int8)

# Align Lags
for col in lag_features:
    if col not in test_df.columns:
        test_df[col] = 0

# Test evaluation logic: Did they acquire a NEW product?
for col in macro_targets:
    test_df[f'{col}_added'] = ((test_df[col] == 1) & (test_df[f'{col}_lag_1'] == 0)).astype(np.int8)

delta_targets = [f'{col}_added' for col in macro_targets]
y_external_test = test_df[delta_targets] 

test_df['total_portfolio_size_lag_1'] = test_df['target_core_activity_lag_1'] + test_df['target_credit_spending_lag_1'] + test_df['target_wealth_and_other_lag_1']
test_df['portfolio_growth'] = 0 # Default test set growth to 0 if lag_2 unavailable
test_df['income_per_product'] = test_df['renta'] / test_df['total_portfolio_size_lag_1'].replace(0, 1)

X_external = test_df[base_features].fillna(0)

# 4. APK Ranking Calculation Helper
def apk(actual, predicted, k=5):
    if len(predicted) > k: predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual: return 0.0
    return score / min(len(actual), k)

# 5. Inference & KAGGLE POST-PROCESSING HACK
print("--- Running Inference & Post-Processing Rules ---")
final_probs = ovr_clf.predict_proba(X_external)

# The Post-Processing Hack: Extract what they ALREADY owned in the previous month
lag_cols = ['target_core_activity_lag_1', 'target_credit_spending_lag_1', 'target_wealth_and_other_lag_1']
lag_matrix = test_df[lag_cols].values

# Force probability to 0 if the user already owns the product
final_probs_adjusted = final_probs * (1 - lag_matrix)

map_actual_ext = []
map_predicted_ext = []
y_external_array = y_external_test.values

for i in range(len(y_external_array)):
    actual_purchases = list(np.where(y_external_array[i] == 1)[0])
    # Rank using the adjusted probabilities
    predicted_ranking = list(np.argsort(final_probs_adjusted[i])[::-1][:5])
    
    map_actual_ext.append(actual_purchases) if actual_purchases else map_actual_ext.append([])
    map_predicted_ext.append(predicted_ranking)

apk_scores_k5 = [apk(a, p, 5) for a, p in zip(map_actual_ext, map_predicted_ext)]
external_map5 = np.mean(apk_scores_k5)

# 6. Export individual APK scores
print("--- Exporting Individual APK@5 Scores to CSV ---")
apk_export_df = pd.DataFrame({
    'ncodpers': test_df['ncodpers'],
    'v3_rf_apk_at_5': apk_scores_k5
})
apk_export_path = r"C:\ML2_final_project\v3_rf_individual_apk_scores_k5.csv"
apk_export_df.to_csv(apk_export_path, index=False)

# 7. Standard Metrics
y_external_pred_binary = ovr_clf.predict(X_external)
ext_acc = accuracy_score(y_external_test, y_external_pred_binary)
ext_mae = mean_absolute_error(y_external_test, y_external_pred_binary)

print("\n" + "="*70)
print("FINAL EXTERNAL SOTA METRICS (V3 PIPELINE)")
print("="*70)
print(f"External MAP@5     : {external_map5:.4f} <-- YOUR FINAL BENCHMARK")
print(f"External Accuracy  : {ext_acc:.4f}")
print(f"External MAE       : {ext_mae:.4f}")
print(f"-> Individual APK@5 scores exported to: {apk_export_path}")
print("="*70)