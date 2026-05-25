# %% [markdown]
# # Phase 4A - Part 1: KNN Benchmark (Distance Baseline)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import gc

print("="*60)
print("BENCHMARK 1: K-NEAREST NEIGHBORS (500k SAMPLE)")
print("="*60)

# Capped sample to prevent O(n*d) memory crash
df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet").sample(n=500000, random_state=42)

target = 'ind_tjcr_fin_ult1'
y = df[target].fillna(0).astype('int8')

feature_cols = ['age', 'renta'] + [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1') and c != target]
X_raw = df[feature_cols].fillna(0)

# KNN requires Z-Score Scaling
print("Standardizing feature space...")
X_scaled = StandardScaler().fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print("Training KNN (n_neighbors=5)...")
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\n--- KNN RESULTS ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"Time     : {time.time() - start_time:.2f} seconds")

del df, X_raw, X_scaled
gc.collect()