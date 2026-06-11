# ==============================================================================
# PHASE 4: GLOBAL OOT PIPELINE (OUT-OF-TIME VALIDATION)
# Architecture: 47D Global PCA -> 24 Parallel Independent Random Forests -> Lag Mask Pruning
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import time
import gc

print("="*80)
print("🚀 INIT: OUT-OF-TIME PIPELINE WITH 24 PARALLEL ENSEMBLES")
print("="*80)
start_time = time.time()

product_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

# --- [1/6] Loading Datasets ---
print("--- [1/6] Ingesting Train & Test Files ---")
train_path = r"C:\ML2_final_project\train_cleaned.parquet"
test_path = r"C:\ML2_final_project\test_cleaned.parquet"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Tag files before combining to ensure strict OOT separation later
df_train['is_test'] = 0
df_test['is_test'] = 1

# Combine temporarily to ensure chronological lag mathematics work perfectly across the split
df = pd.concat([df_train, df_test], ignore_index=True)
df = df.sort_values(by=['ncodpers', 'fecha_dato'])

del df_train, df_test; gc.collect()

# --- [2/6] Sanitization & Temporal Engineering ---
print("--- [2/6] Engineering Temporal Lags & Delta Targets ---")
numeric_cols = ['age', 'renta', 'antiguedad']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = np.nan

# Global 3-Sigma Winsorization for Income
if 'renta' in df.columns:
    upper_bound = df['renta'].mean() + (3 * df['renta'].std())
    df['renta'] = np.clip(df['renta'], a_min=None, a_max=upper_bound)

delta_targets, lag_features = [], []
for col in product_cols:
    if col in df.columns:
        lag_col = f'{col}_lag_1'
        # Shift correctly across the chronological dataset
        df[lag_col] = df.groupby('ncodpers')[col].shift(1).fillna(0).astype(np.int8)
        lag_features.append(lag_col)
        
        added_col = f'{col}_added'
        df[added_col] = ((df[col] == 1) & (df[lag_col] == 0)).astype(np.int8)
        delta_targets.append(added_col)

cols_to_drop = ['tipodom', 'conyuemp', 'ult_fec_cli_1t']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

if 'fecha_dato' in df.columns:
    df['month'] = pd.to_datetime(df['fecha_dato']).dt.month.astype(np.int8)

df['total_products_owned'] = df[lag_features].sum(axis=1).astype(np.int8)

if 'canal_entrada' in df.columns:
    top_5_channels = df['canal_entrada'].value_counts().nlargest(5).index.tolist()
    df['canal_entrada_grouped'] = df['canal_entrada'].apply(lambda x: x if x in top_5_channels else 'OTHER')
    df['canal_entrada_grouped'] = df['canal_entrada_grouped'].astype('category').cat.codes
    df.drop(columns=['canal_entrada'], inplace=True)

df.dropna(subset=lag_features, inplace=True)

# --- [3/6] Strict OOT Split & PCA ---
print("--- [3/6] Executing Strict Chronological Split & Global PCA ---")
leakage_cols = [c for c in df.columns if c in product_cols or c in delta_targets]

# Split back to isolated environments
train_env = df[df['is_test'] == 0].copy()
test_env = df[df['is_test'] == 1].copy()
del df; gc.collect()

X_train = train_env.drop(columns=leakage_cols + ['is_test', 'fecha_dato'])
y_train = train_env[delta_targets]

X_test = test_env.drop(columns=leakage_cols + ['is_test', 'fecha_dato'])
y_test = test_env[delta_targets]

# Standardize formats
for col in X_train.columns:
    if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes
    fill_val = X_train[col].median() if col in numeric_cols else 0
    X_train[col] = X_train[col].fillna(fill_val)
    X_test[col] = X_test[col].fillna(fill_val)

# PCA Compression (Fitted strictly on Train, applied to Test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_final = pca.fit_transform(X_train_scaled)
X_test_final = pca.transform(X_test_scaled)

print(f"✔️ Train Set: {X_train_final.shape[0]} rows | Test Set: {X_test_final.shape[0]} rows")
print(f"✔️ Global PCA compressed {X_train.shape[1]} features into {X_train_final.shape[1]} Principal Components.")

del X_train, X_test, X_train_scaled, X_test_scaled; gc.collect()

# --- [4/6] Parallel Loop Ensemble Training ---
print("--- [4/6] Initiating 24 Parallel Loops: Target-Specific RUS & Random Forests ---")
predictions_list = []

for idx, target_col in enumerate(delta_targets):
    y_train_single = y_train[target_col].values
    positives = y_train_single.sum()
    
    if positives < 50: # Skip dead products with no momentum
        predictions_list.append(np.zeros(len(X_test_final)))
        continue
        
    print(f" -> Loop {idx+1:02d}/24: Learning [{target_col}] | Actual Buyers: {positives}")
    
    # Target-Specific Bias Destruction
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train_final, y_train_single)
    
    # The Independent Core
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_resampled, y_resampled)
    
    predictions_list.append(rf.predict_proba(X_test_final)[:, 1])

raw_probs = np.column_stack(predictions_list)

# --- [5/6] Post-Processing: Ownership Lag Mask ---
print("--- [5/6] Applying Ownership Lag Mask Pruning ---")
lag_matrix = test_env[lag_features].values
final_probs_adjusted = raw_probs * (1 - lag_matrix) # Zeroes out already-owned items

top5_indices = np.argsort(final_probs_adjusted, axis=1)[:, -5:][:, ::-1]
final_top5_recommendations = [[product_cols[idx] for idx in user_row] for user_row in top5_indices]

# --- [6/6] MAP@5 Evaluation ---
print("--- [6/6] Calculating MAP@5 Evaluation Metric ---")
def apk(actual, predicted, k=5):
    if len(predicted) > k: predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual: return 0.0
    return score / min(len(actual), k)

actual_purchases = []
y_test_matrix = y_test.values

for i in range(len(y_test_matrix)):
    bought_indices = np.where(y_test_matrix[i] == 1)[0]
    bought_items = [product_cols[idx] for idx in bought_indices]
    actual_purchases.append(bought_items)

map_scores = [apk(actual_purchases[i], final_top5_recommendations[i], k=5) for i in range(len(actual_purchases)) if len(actual_purchases[i]) > 0]

print("\n" + "="*80)
print(f"🎯 OOT VALIDATION MAP@5 SCORE: {np.mean(map_scores):.4f}")
print(f"⏱️ TOTAL EXECUTION TIME: {(time.time() - start_time)/60:.2f} Minutes")
print("="*80)
