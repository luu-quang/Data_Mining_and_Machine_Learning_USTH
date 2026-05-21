# %% [markdown]
# # Phase 5: SOTA Hyperparameter Tuning
# ---------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (Model Evaluation):
1. Why do we need this script? We can't just guess that 150 trees at depth 10 is the best.
   The professor will ask "How do you know?"
2. RandomizedSearchCV: This tests random combinations of parameters and uses 
   Cross-Validation (splitting the training data 3 times to double-check itself).
3. ROC-AUC: We optimize the grid search for ROC-AUC (Area Under the ROC Curve) 
   instead of accuracy, because we care about the quality of the PROBABILITY rankings, 
   not just binary 0s and 1s.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import gc

print("="*60)
print("PHASE 5: HYPERPARAMETER TUNING (PCA FUSION ARCHITECTURE)")
print("="*60)

df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet").sample(n=50000, random_state=42)

# Lags & PCA Setup (Replicating Phase 4 environment)
df['lag_1'] = df['ind_cco_fin_ult1'].shift(1).fillna(0)
df['lag_2'] = df['ind_cco_fin_ult1'].shift(2).fillna(0)
df['lag_3'] = df['ind_cco_fin_ult1'].shift(3).fillna(0)
df['momentum'] = df['lag_1'] + df['lag_2'] + df['lag_3']

base_features = ['age', 'renta'] + [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1')]
df_pca_safe = df[base_features].fillna(0) 
X_base_scaled = StandardScaler().fit_transform(df_pca_safe)

pca = PCA(n_components=19)
pca_features = pca.fit_transform(X_base_scaled)
pca_df = pd.DataFrame(pca_features, index=df.index, columns=[f'PC_{i}' for i in range(1, 20)])

X = pd.concat([df[['age', 'renta', 'lag_1', 'lag_2', 'lag_3', 'momentum']], pca_df], axis=1)

# Proxy Tuning: We tune on the hardest target (Credit Cards) to find the best tree structure
target = 'ind_tjcr_fin_ult1'
y = df[target].fillna(0).astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n--- Executing Randomized Hyperparameter Search ---")
param_grid = {
    'n_estimators': [50, 100, 150],       
    'max_depth': [4, 6, 8, 10],           
    'min_samples_split': [2, 5, 10],      
    'class_weight': ['balanced']          
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# The Engine: 5 iterations x 3-Fold CV = 15 different models tested automatically
rf_tuned = RandomizedSearchCV(
    estimator=base_rf, 
    param_distributions=param_grid, 
    n_iter=5,             
    cv=3,                 
    scoring='roc_auc',    
    random_state=42,
    verbose=1
)

start_time = time.time()
rf_tuned.fit(X_train, y_train)

print(f"\n  -> Tuning Execution Time: {time.time() - start_time:.2f} seconds")
print("\n" + "="*60)
print("[OPTIMAL TREE ARCHITECTURE DISCOVERED]")
print("="*60)
for param, value in rf_tuned.best_params_.items():
    print(f"{param}: {value}")

del df, df_pca_safe, X, y
gc.collect()