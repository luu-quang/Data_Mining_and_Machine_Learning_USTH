# %% [markdown]
# # V2 Architecture: Hierarchical Macro-Category Classification (Affinity-Optimized)
# ----------------------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (V2 Architecture):
1. Data-Driven Macro-Aggregation: We grouped 24 products into 3 dense clusters 
   based on Pearson Correlation (Affinity) rather than manual business logic.
2. Feature Engineering: Added demographic context (Relative Income) and historical 
   engagement (Total Portfolio Size) to maximize Gini Importance.
3. Natural Prior Subsampling: We train on a random 2.5M row slice to preserve the 
   true market distribution, preventing the probability drift seen in 50/50 splits.
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
print("V2 PIPELINE: AFFINITY-BASED MACRO-CATEGORY CLASSIFICATION")
print("="*70)

# [1] Define the Macro-Category Dictionaries (Optimized via Correlation Heatmap)
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

# [3] Feature Engineering: Context & Aggregations
print("--- Engineering Contextual Features & Demographics ---")

# FIX: Force messy Santander columns to true numeric types
numeric_cols = ['age', 'renta', 'antiguedad', 'ind_actividad_cliente']
for col in numeric_cols:
    if col in df.columns:
        # 'coerce' turns text strings like "  NA" into true NaNs
        df[col] = pd.to_numeric(df[col], errors='coerce')

# A. Encode Categoricals safely
cat_cols = ['sexo', 'segmento', 'nomprov']
le_dict = {}

# A. Encode Categoricals safely
cat_cols = ['sexo', 'segmento', 'nomprov']
le_dict = {}

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('object')
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# B. Engineer Relative Income (Socio-Economic Context)
if 'renta' in df.columns and 'nomprov' in df.columns:
    prov_median_renta = df.groupby('nomprov')['renta'].transform('median')
    df['renta_relative'] = df['renta'] / prov_median_renta.replace(0, 1)

# C. Engineer Portfolio Size (Historical Engagement)
product_cols = [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1')]
df['total_portfolio_size'] = df[product_cols].sum(axis=1)

# [4] Engineer Macro-Targets & Temporal Lags
print("--- Aggregating Clusters & Engineering Momentum Lags ---")
for macro_name, sub_cols in macro_mapping.items():
    valid_cols = [c for c in sub_cols if c in df.columns]
    df[f'target_{macro_name.lower()}'] = (df[valid_cols].sum(axis=1) > 0).astype(np.int8)

lag_features = []
for col in macro_targets:
    df[f'{col}_lag_1'] = df.groupby('ncodpers')[col].shift(1).fillna(0)
    df[f'{col}_lag_2'] = df.groupby('ncodpers')[col].shift(2).fillna(0)
    df[f'{col}_lag_3'] = df.groupby('ncodpers')[col].shift(3).fillna(0)
    df[f'{col}_momentum'] = df[f'{col}_lag_1'] + df[f'{col}_lag_2'] + df[f'{col}_lag_3']
    lag_features.extend([f'{col}_lag_1', f'{col}_lag_2', f'{col}_lag_3', f'{col}_momentum'])

# Combine Final Feature Space
demographics = ['age', 'renta', 'antiguedad', 'ind_actividad_cliente', 'sexo', 'segmento', 'nomprov']
aggregations = ['renta_relative', 'total_portfolio_size']
base_features = [c for c in (demographics + aggregations + lag_features) if c in df.columns]

X_raw = df[base_features].fillna(0)
y_raw = df[macro_targets]

del df
gc.collect()

# [5] Train/Test Split & Natural Subsampling
print("--- Splitting and Executing Natural Subsampling (2.5M Rows) ---")
X_train_raw, X_test_final, y_train, y_test_final = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

# Extract 2.5M rows preserving natural market distribution
SAMPLE_SIZE = 2500000
X_train_final = X_train_raw.sample(n=SAMPLE_SIZE, random_state=42)
y_train_final = y_train.loc[X_train_final.index]

del X_train_raw, X_raw, y_raw, y_train
gc.collect()

# [6] Train One-Vs-Rest Random Forest
print("\n--- Training V2 Macro Random Forest Ensemble ---")
start_time = time.time()

tuned_rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=10, 
    min_samples_split=10, 
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
print("INTERNAL VALIDATION (MACRO-INTENT PREDICTION)")
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

# FIX: Clean the test data numerics too
for col in numeric_cols:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# 1. Apply Categorical Encodings
for col in cat_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('object')
        test_df[col] = test_df[col].fillna('UNKNOWN')
        classes = le_dict[col].classes_
        test_df[col] = test_df[col].astype(str).apply(lambda x: x if x in classes else 'UNKNOWN')
        test_df[col] = le_dict[col].transform(test_df[col])

# 2. Engineer Target Aggregations & Features for External Set
if 'renta' in test_df.columns and 'nomprov' in test_df.columns:
    prov_median_renta_test = test_df.groupby('nomprov')['renta'].transform('median')
    test_df['renta_relative'] = test_df['renta'] / prov_median_renta_test.replace(0, 1)

test_df['total_portfolio_size'] = test_df[[c for c in test_df.columns if c in product_cols]].sum(axis=1)

for macro_name, sub_cols in macro_mapping.items():
    valid_cols = [c for c in sub_cols if c in test_df.columns]
    test_df[f'target_{macro_name.lower()}'] = (test_df[valid_cols].sum(axis=1) > 0).astype(np.int8)

y_external_test = test_df[macro_targets]

for col in lag_features:
    if col not in test_df.columns:
        test_df[col] = 0

X_external = test_df[base_features].fillna(0)

# 3. APK Ranking Calculation Helper
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

# 4. Inference & Ranking
final_probs = ovr_clf.predict_proba(X_external)

map_actual_ext = []
map_predicted_ext = []
y_external_array = y_external_test.values

for i in range(len(y_external_array)):
    actual_purchases = list(np.where(y_external_array[i] == 1)[0])
    predicted_ranking = list(np.argsort(final_probs[i])[::-1][:5])
    map_actual_ext.append(actual_purchases) if actual_purchases else map_actual_ext.append([])
    map_predicted_ext.append(predicted_ranking)

apk_scores_k5 = [apk(a, p, 5) for a, p in zip(map_actual_ext, map_predicted_ext)]
external_map5 = np.mean(apk_scores_k5)

# 5. Export individual APK scores
print("--- Exporting Individual APK@5 Scores to CSV ---")
apk_export_df = pd.DataFrame({
    'ncodpers': test_df['ncodpers'],
    'v2_rf_apk_at_5': apk_scores_k5
})
apk_export_path = r"C:\ML2_final_project\v2_rf_individual_apk_scores_k5.csv"
apk_export_df.to_csv(apk_export_path, index=False)

# 6. Standard Metrics
y_external_pred_binary = ovr_clf.predict(X_external)

ext_acc = accuracy_score(y_external_test, y_external_pred_binary)
ext_mae = mean_absolute_error(y_external_test, y_external_pred_binary)

print("\n" + "="*70)
print("FINAL EXTERNAL SOTA METRICS (AFFINITY CLUSTERS)")
print("="*70)
print(f"External MAP@5     : {external_map5:.4f} <-- THE V2 BENCHMARK")
print(f"External Accuracy  : {ext_acc:.4f}")
print(f"External MAE       : {ext_mae:.4f}")
print(f"-> Individual APK@5 scores exported to: {apk_export_path}")
print("="*70)