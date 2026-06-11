# ==============================================================================
# PIPELINE (MODIFIED): PCA APPLIED TO ALL FEATURES
# Warning: Applying PCA to sparse binary features degrades Random Forest performance.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import time
import gc

print("="*80)
print("🚀 INIT: PIPELINE WITH GLOBAL PCA (APPLIED TO ALL FEATURES)")
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

print("--- [1/6] Loading & Sanitizing Dataset ---")
train_path = r"C:\ML2_final_project\train_cleaned.parquet" 
df = pd.read_parquet(train_path)
df = df.sort_values(by=['ncodpers', 'fecha_dato'])

numeric_cols = ['age', 'renta', 'antiguedad']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = np.nan

if 'renta' in df.columns:
    upper_bound = df['renta'].mean() + (3 * df['renta'].std())
    df['renta'] = np.clip(df['renta'], a_min=None, a_max=upper_bound)

print("--- [2/6] Engineering Temporal Lags & Targets ---")
delta_targets, lag_features = [], []
for col in product_cols:
    if col in df.columns:
        lag_col = f'{col}_lag_1'
        df[lag_col] = df.groupby('ncodpers')[col].shift(1).fillna(0).astype(np.int8)
        lag_features.append(lag_col)
        
        added_col = f'{col}_added'
        df[added_col] = ((df[col] == 1) & (df[lag_col] == 0)).astype(np.int8)
        delta_targets.append(added_col)

# Extract Latent Features
cols_to_drop = ['tipodom', 'conyuemp', 'ult_fec_cli_1t']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

if 'fecha_dato' in df.columns:
    df['month'] = pd.to_datetime(df['fecha_dato']).dt.month.astype(np.int8)

df['total_products_owned'] = df[lag_features].sum(axis=1).astype(np.int8)

if 'canal_entrada' in df.columns:
    top_5_channels = df['canal_entrada'].value_counts().nlargest(5).index.tolist()
    df['canal_entrada_grouped'] = df['canal_entrada'].apply(lambda x: x if x in top_5_channels else 'OTHER')
    df['canal_entrada_grouped'] = df['canal_entrada_grouped'].astype('category').cat.codes
    df.drop(columns=['canal_entrada'], inplace=True)

df.dropna(subset=lag_features, inplace=True)

print("--- [3/6] Train/Test Split & GLOBAL PCA ---")
leakage_cols = [c for c in df.columns if c in product_cols or c in delta_targets]
X_raw = df.drop(columns=leakage_cols)
y_raw = df[delta_targets]

# Convert all categoricals to numeric codes
for col in X_raw.columns:
    if X_raw[col].dtype.name == 'category' or X_raw[col].dtype == 'object':
        X_raw[col] = X_raw[col].astype('category').cat.codes
    X_raw[col] = X_raw[col].fillna(X_raw[col].median() if col in numeric_cols else 0)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

# APPLYING PCA TO EVERY SINGLE FEATURE (Lags, Month, Demographics)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_final = pca.fit_transform(X_train_scaled)
X_test_final = pca.transform(X_test_scaled)

print(f"Global PCA reduced feature space from {X_train.shape[1]} to {X_train_final.shape[1]} components.")

del X_train, X_test, X_raw; gc.collect()

print("--- [4/6] Ensemble Training (Expect longer training times) ---")
predictions_list = []

for idx, target_col in enumerate(delta_targets):
    y_train_single = y_train[target_col].values
    positives = y_train_single.sum()
    
    if positives < 50:
        predictions_list.append(np.zeros(len(X_test_final)))
        continue
        
    print(f" -> Training Model {idx+1:02d}/24: {target_col} | Positives: {positives}")
    
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train_final, y_train_single)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_resampled, y_resampled)
    
    predictions_list.append(rf.predict_proba(X_test_final)[:, 1])

raw_probs = np.column_stack(predictions_list)

print("--- [5/6] Post-Processing: Lag Mask ---")
lag_matrix = df.loc[y_test.index, lag_features].values
final_probs_adjusted = raw_probs * (1 - lag_matrix)

top5_indices = np.argsort(final_probs_adjusted, axis=1)[:, -5:][:, ::-1]
final_top5_recommendations = [[product_cols[idx] for idx in user_row] for user_row in top5_indices]

print("--- [6/6] Calculating MAP@5 Metrics ---")
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
print(f"🎯 NEW METRIC - MAP@5 SCORE: {np.mean(map_scores):.4f}")
print("="*80)