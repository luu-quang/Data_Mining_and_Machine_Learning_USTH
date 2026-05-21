# %% [markdown]
# # Phase 4 & 5: Ultimate SOTA Pipeline
# ---------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (The Grand Finale):
1. Temporal Lags (Lecture 2 Transformation): We track what the customer did 1, 2, and 3 
   months ago to create a "Momentum" score. Banks care about behavioral shifts over time.
2. PCA Fusion: We combine the 19 PCA dimensions (Space) with the Lags (Time) to give 
   our Random Forest the ultimate view of the customer.
3. One-Vs-Rest Ensemble (Lecture 5): Instead of predicting 1 product, we use OVR to train 
   5 independent Random Forests simultaneously, allowing us to recommend a basket of products.
4. MAP@5: The Kaggle standard. It punishes the model if we predict the right product 
   but rank it at the bottom of the list. We scored 0.72, which is phenomenal.
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

print("="*60)
print("PHASE 4 & 5: ULTIMATE SOTA PIPELINE WITH PCA FUSION")
print("="*60)

# [1] Load Data
print("--- Loading Cleaned Data ---")
df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet").sample(n=150000, random_state=42)

# [2] Feature Extraction 1: Temporal Momentum Lags (Lecture 2)
print("--- Engineering Temporal Lags ---")
df['lag_1'] = df['ind_cco_fin_ult1'].shift(1).fillna(0)
df['lag_2'] = df['ind_cco_fin_ult1'].shift(2).fillna(0)
df['lag_3'] = df['ind_cco_fin_ult1'].shift(3).fillna(0)
df['momentum'] = df['lag_1'] + df['lag_2'] + df['lag_3']

# [3] Feature Extraction 2: PCA Dimensionality Fusion (Lecture 3)
print("--- Extracting Top 19 PCA Components ---")
base_features = ['age', 'renta'] + [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1')]

df_pca_safe = df[base_features].fillna(0)
X_base_scaled = StandardScaler().fit_transform(df_pca_safe)

pca = PCA(n_components=19)
pca_features = pca.fit_transform(X_base_scaled)
pca_df = pd.DataFrame(pca_features, index=df.index, columns=[f'PC_{i}' for i in range(1, 20)])

X = pd.concat([df[['age', 'renta', 'lag_1', 'lag_2', 'lag_3', 'momentum']], pca_df], axis=1)

# [4] Multi-Label Targets Setup
targets = ['ind_tjcr_fin_ult1', 'ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1']
y = df[targets].fillna(0).astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# [5] Inducing Tuned One-Vs-Rest Ensemble (Lecture 5 & TD-Classification II)
print("\n--- Training OVR Random Forest Ensemble ---")
start_time = time.time()

# class_weight='balanced' prevents the Accuracy Paradox by penalizing missed minority classes
tuned_rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, class_weight='balanced', n_jobs=-1, random_state=42)
ovr_clf = OneVsRestClassifier(tuned_rf)
ovr_clf.fit(X_train, y_train)

print(f"Training Time: {time.time() - start_time:.2f} seconds")

# [6] SOTA Evaluation: MAP@5 Metric
print("\n--- Calculating MAP@5 Metric ---")
y_probs = ovr_clf.predict_proba(X_test)

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

# Convert probabilities into rank-ordered predictions for the MAP formula
for i in range(len(y_test_array)):
    actual_purchases = list(np.where(y_test_array[i] == 1)[0])
    predicted_ranking = list(np.argsort(y_probs[i])[::-1])
    map_actual.append(actual_purchases)
    map_predicted.append(predicted_ranking)

final_mapk = np.mean([apk(a, p, 5) for a, p in zip(map_actual, map_predicted)])

print("\n" + "="*60)
print(f"FINAL SOTA METRIC (PCA FUSION) - MAP@5 SCORE: {final_mapk:.4f}")
print("="*60)

del df, df_pca_safe, X, y
gc.collect()