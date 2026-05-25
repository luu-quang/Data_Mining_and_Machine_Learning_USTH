# %% [markdown]
# # Phase 4A - Part 2: Decision Tree Benchmark (Full Data)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import gc

print("="*60)
print("BENCHMARK 2: DECISION TREE (FULL DATASET)")
print("="*60)

print("Loading full dataset...")
df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet")

target = 'ind_tjcr_fin_ult1'
y = df[target].fillna(0).astype('int8')

feature_cols = ['age', 'renta'] + [c for c in df.columns if c.startswith('ind_') and c.endswith('ult1') and c != target]
# No scaling needed for Tree models!
X = df[feature_cols].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training Decision Tree (class_weight='balanced')...")
start_time = time.time()
dt = DecisionTreeClassifier(class_weight='balanced', max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(f"\n--- DECISION TREE RESULTS ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"Time     : {time.time() - start_time:.2f} seconds")

del df, X
gc.collect()