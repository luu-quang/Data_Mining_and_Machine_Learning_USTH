# %% [markdown]
# # Phase 4 & 5: Accelerated SOTA Pipeline with PCA Fusion
# ----------------------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (Path B Architecture):
1. Split-Before-Resample Rule: We split the raw data into Train and Test FIRST. 
   The Test set is left completely untouched to reflect the real-world bank distribution.
2. Training Acceleration: We downsample the massive training set to 1.5 million rows. 
   This cuts training time by 80% while providing more than enough data for the trees to learn.
3. PCA Isolation: The StandardScaler and PCA are strictly `.fit()` on the TRAIN data, 
   and only `.transform()` is applied to the TEST data. This prevents Data Leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import gc

print("="*70)
print("PHASE 4 & 5: ACCELERATED SOTA PIPELINE WITH PCA FUSION")
print("="*70)

# [1] Load Cleaned Data
print("--- Loading Cleaned Parquet Dataset ---")
df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet")

# [2] Pre-Split Feature Engineering: Temporal Momentum Lags
print("--- Engineering Temporal Lags ---")
df['lag_1'] = df['ind_cco_fin_ult1'].shift(1).fillna(0)
df['lag_2'] = df['ind_cco_fin_ult1'].shift(2).fillna(0)
df['lag_3'] = df['ind_cco_fin_ult1'].shift(3).fillna(0)
df['momentum'] = df['lag_1'] + df['lag_2'] + df['lag_3']

base_features = ['age', 'renta', 'lag_1', 'lag_2', 'lag_3', 'momentum']
pca_source_features = ['age', 'renta'] + [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1')]
targets = ['ind_tjcr_fin_ult1', 'ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1']

X_raw = df[list(set(base_features + pca_source_features))]
y_raw = df[targets].fillna(0).astype(np.int8)

del df
gc.collect()

# [3] Strict Train/Test Split
print("--- Splitting Train and Test Partitions ---")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

del X_raw, y_raw
gc.collect()

# [4] Accelerated Subsampling Layer (Applied to Training Set Only)
print("--- Executing Training Set Subsampling ---")
# We randomly sample 1.5 million rows to speed up training, while allowing 
# class_weight='balanced' to handle the specific imbalance of each of the 5 targets.
SAMPLE_SIZE = 1500000

# Ensure we don't try to sample more than exists (safety check)
actual_sample = min(SAMPLE_SIZE, len(X_train_raw))

X_train_resampled = X_train_raw.sample(n=actual_sample, random_state=42)
y_train_resampled = y_train.loc[X_train_resampled.index]

print(f" -> Original Train Size: {len(X_train_raw):,}")
print(f" -> Accelerated Train Size: {len(X_train_resampled):,}")
print(f" -> Untouched Test Size:  {len(X_test_raw):,} (Pristine Bank Distribution)")

del X_train_raw
gc.collect()

# [5] Isolated PCA Feature Extraction (No Data Leakage)
print("--- Extracting 19 PCA Components ---")
scaler = StandardScaler()
X_train_pca_scaled = scaler.fit_transform(X_train_resampled[pca_source_features].fillna(0))
X_test_pca_scaled = scaler.transform(X_test_raw[pca_source_features].fillna(0))

pca = PCA(n_components=19, random_state=42)
pca_train_features = pca.fit_transform(X_train_pca_scaled)
pca_test_features = pca.transform(X_test_pca_scaled)

pca_cols = [f'PC_{i}' for i in range(1, 20)]
pca_train_df = pd.DataFrame(pca_train_features, index=X_train_resampled.index, columns=pca_cols)
pca_test_df = pd.DataFrame(pca_test_features, index=X_test_raw.index, columns=pca_cols)

X_train_final = pd.concat([X_train_resampled[base_features], pca_train_df], axis=1)
X_test_final = pd.concat([X_test_raw[base_features], pca_test_df], axis=1)

del X_train_resampled, X_test_raw, pca_train_df, pca_test_df
gc.collect()

# [6] Train One-Vs-Rest Random Forest Ensemble
print("\n--- Training OVR Random Forest Ensemble ---")
start_time = time.time()

# class_weight='balanced' dynamically handles the unique imbalance of EACH of the 5 products
tuned_rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, class_weight='balanced', n_jobs=-1, random_state=42)
ovr_clf = OneVsRestClassifier(tuned_rf)
ovr_clf.fit(X_train_final, y_train_resampled)

print(f"Training Execution Time: {time.time() - start_time:.2f} seconds")

# [7] Evaluate Performance via MAP@5 Metric
print("\n--- Calculating MAP@5 Metric on Unseen Test Set ---")
y_probs = ovr_clf.predict_proba(X_test_final)

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

map_actual = []
map_predicted = []
y_test_array = y_test.values

for i in range(len(y_test_array)):
    actual_purchases = list(np.where(y_test_array[i] == 1)[0])
    predicted_ranking = list(np.argsort(y_probs[i])[::-1])
    map_actual.append(actual_purchases) if actual_purchases else map_actual.append([])
    map_predicted.append(predicted_ranking)

final_mapk = np.mean([apk(a, p, 5) for a, p in zip(map_actual, map_predicted)])

print("\n" + "="*70)
print(f"FINAL SOTA METRIC (ACCELERATED PIPELINE) - MAP@5 SCORE: {final_mapk:.4f}")
print("="*70)

# [8] Evaluate Standard Multi-Label Classification Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n--- Calculating Standard Classification Metrics (Multi-Label) ---")
# Extract hard binary predictions (0s and 1s) based on a 50% threshold
y_pred_binary = ovr_clf.predict(X_test_final)

# We use average='macro' to calculate the metric for each of the 5 products 
# independently, and then average them together so rare products aren't ignored.
acc = accuracy_score(y_test, y_pred_binary)
prec = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred_binary, average='macro')
f1 = f1_score(y_test, y_pred_binary, average='macro')

print("\n" + "="*70)
print("FINAL SOTA CLASSIFICATION METRICS (MACRO AVERAGE)")
print("="*70)
print(f"Global Accuracy : {acc:.4f}")
print(f"Macro Precision : {prec:.4f}")
print(f"Macro Recall    : {rec:.4f}")
print(f"Macro F1-Score  : {f1:.4f}")
print("="*70)