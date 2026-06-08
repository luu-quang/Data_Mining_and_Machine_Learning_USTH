# %% [markdown]
# # Final SOTA Architecture: Hierarchical Macro-Category Acquisition Model
# # KDD Pipeline: Recommendation Engine (Santander Dataset)
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import time
import gc

print("="*75)
print("FINAL PIPELINE: OPTIMIZED MACRO-CATEGORY RECOMMENDATION ENGINE")
print("="*75)
start_time = time.time()

# [1] The Mathematically Validated Groupings
macro_mapping = {
    'Transactional': ['ind_cco_fin_ult1', 'ind_ecue_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctju_fin_ult1', 'ind_cder_fin_ult1', 'ind_recibo_ult1'],
    'Accumulation': ['ind_ahor_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_fond_fin_ult1', 'ind_valo_fin_ult1', 'ind_plan_fin_ult1'],
    'Credit_Leverage': ['ind_tjcr_fin_ult1', 'ind_pres_fin_ult1', 'ind_hip_fin_ult1'],
    'General_Specialized': ['ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_cno_fin_ult1', 'ind_aval_fin_ult1', 'ind_reca_fin_ult1', 'ind_viv_fin_ult1']
}
macro_targets = [f'target_{name.lower()}' for name in macro_mapping.keys()]

# [2] Load Cleaned Data (Fast Parquet)
print("--- [1/5] Loading Parquet Dataset ---")
train_path = r"C:\ML2_final_project\train_cleaned.parquet" 
df = pd.read_parquet(train_path)

# OPTIMIZATION: Sort by user and date to ensure temporal logic works perfectly
df = df.sort_values(by=['ncodpers', 'fecha_dato'])

# [3] Vectorized Feature Engineering & Target Generation
print("--- [2/5] Engineering Temporal Features & Matrix Deltas ---")

# Convert Demographics efficiently
numeric_cols = ['age', 'renta', 'antiguedad']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

# Generate Macro-Targets (Vectorized with int8 to save massive amounts of RAM)
for macro_name, sub_cols in macro_mapping.items():
    valid_cols = [c for c in sub_cols if c in df.columns]
    df[f'target_{macro_name.lower()}'] = (df[valid_cols].sum(axis=1) > 0).astype(np.int8)

# Calculate Lags and Acquisition Deltas
delta_targets = []
lag_features = []

for col in macro_targets:
    lag_col = f'{col}_lag_1'
    df[lag_col] = df.groupby('ncodpers')[col].shift(1).fillna(0).astype(np.int8)
    lag_features.append(lag_col)
    
    # Core Logic: We only predict ACQUISITIONS (Owned 0 in T-1, but Owned 1 in T)
    added_col = f'{col}_added'
    df[added_col] = ((df[col] == 1) & (df[lag_col] == 0)).astype(np.int8)
    delta_targets.append(added_col)

df.dropna(subset=lag_features, inplace=True)
gc.collect()

# [4] Train/Test Split & Feature Formatting
print("--- [3/5] Constructing Feature Space & Handling Missing Data ---")
# Drop all target columns to prevent Data Leakage
leakage_cols = [c for c in df.columns if 'target_' in c or ('ind_' in c and c.endswith('_ult1'))]
X_raw = df.drop(columns=leakage_cols)
y_raw = df[delta_targets]

# BUG FIX: Handle Categorical Strings and NaNs for Scikit-Learn
for col in X_raw.columns:
    if X_raw[col].dtype.name == 'category' or X_raw[col].dtype == 'object':
        # cat.codes converts strings to integers and maps NaNs to -1 (which trees handle well)
        X_raw[col] = X_raw[col].astype('category').cat.codes
    else:
        # Fill numeric NaNs with 0
        X_raw[col] = X_raw[col].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

# [5] Train the SOTA Random Forest
print("--- [4/5] Training Optimized Ensemble (OneVsRest + Balanced RF) ---")
# PARAMETERS:
# n_jobs=-1: Utilizes all CPU cores for speed
# class_weight='balanced': Crucial to penalize the model for ignoring sparse classes
base_rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=20, 
    min_samples_split=20, 
    class_weight='balanced', 
    n_jobs=-1, 
    random_state=42
)

ovr_clf = OneVsRestClassifier(base_rf)
ovr_clf.fit(X_train, y_train)

# [6] Inference & Post-Processing
print("--- [5/5] Inference & MAP@5 Post-Processing Constraints ---")
raw_probs = ovr_clf.predict_proba(X_test)

# BUG FIX: Extract lag_matrix directly from the original 'df' using X_test's index
# This avoids the KeyError while perfectly aligning the rows with the predictions!
lag_matrix = df.loc[X_test.index, lag_features].values
final_probs_adjusted = raw_probs * (1 - lag_matrix)

# [7] Output
print("\n" + "="*75)
print(f"PIPELINE COMPLETE. Total execution time: {(time.time() - start_time)/60:.2f} minutes.")
print("="*75)
print("The Recommendation Engine is now ready for MAP@5 metric evaluation.")

# [8] MAP Evaluation Metric Calculation (DM 4/5: Model Evaluation)
print("--- [6/6] Calculating Mean Average Precision (MAP) ---")

def apk(actual, predicted, k=4):
    """Computes the Average Precision for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    if not actual:
        return 0.0
    return score / min(len(actual), k)

# Get the names of the Macro-Categories
category_names = y_test.columns.tolist()

# Get the top predictions from the adjusted probabilities (final_probs_adjusted)
# np.argsort sorts from lowest to highest, so we use [:, ::-1] to reverse it to descending order
pred_indices = np.argsort(final_probs_adjusted, axis=1)[:, ::-1]

# Convert the actual y_test matrix into a list of purchased categories
y_test_matrix = y_test.values
map_scores = []

for i in range(len(y_test_matrix)):
    # Get the index of the categories the customer ACTUALLY bought (where value == 1)
    actual_indices = np.where(y_test_matrix[i] == 1)[0].tolist()
    
    # Only calculate the score for customers who ACTUALLY made a new purchase
    if len(actual_indices) > 0:
        predicted = pred_indices[i].tolist()
        score = apk(actual_indices, predicted, k=4) # k=4 because we have 4 Macro-Categories
        map_scores.append(score)

# Calculate the mean MAP across the entire test set
final_map_score = np.mean(map_scores)

print("\n" + "="*75)
print(f"FINAL MODEL EVALUATION METRIC")
print(f"MACRO-CATEGORY MAP SCORE: {final_map_score:.5f}")
print("="*75)

# =====================================================================
# [9] STAGE 2: MICRO-PRODUCT RANKING (QUOTA 3-2 STRATEGY)
# =====================================================================
print("\n--- [7/7] Generating Final MAP@5 Micro-Product Recommendations ---")

# 1. Calculate the overall popularity (purchase frequency) of the 24 products in the training set
product_cols = [p for sublist in macro_mapping.values() for p in sublist if p in df.columns]
popularity_series = df[product_cols].sum().sort_values(ascending=False)

# 2. Map the popular products strictly to their Macro-Categories
popular_by_macro = {}
for macro_name, prods in macro_mapping.items():
    valid_prods = [p for p in prods if p in product_cols]
    # Sort the products inside this specific category by their overall popularity
    sorted_prods = popularity_series[valid_prods].sort_values(ascending=False).index.tolist()
    popular_by_macro[macro_name] = sorted_prods

# 3. Apply the 3-2 Quota Allocation for each user in the Test Set
macro_keys = list(macro_mapping.keys()) # ['Transactional', 'Accumulation', 'Credit_Leverage', 'General_Specialized']
final_top5_recommendations = []

for i in range(len(pred_indices)):
    # Get the index of the Top 1 and Top 2 Macro Categories for this specific user
    top_1_macro_idx = pred_indices[i][0]
    top_2_macro_idx = pred_indices[i][1]
    
    # Get the actual string names of those categories
    top_1_macro_name = macro_keys[top_1_macro_idx]
    top_2_macro_name = macro_keys[top_2_macro_idx]
    
    # HEURISTIC ROUTING: Quota 3 - 2
    # Take Top 3 most popular products from the 1st Category
    user_recs = popular_by_macro[top_1_macro_name][:3]
    
    # Take Top 2 most popular products from the 2nd Category
    user_recs += popular_by_macro[top_2_macro_name][:2]
    
    # Fallback: In case a category has fewer than 2 or 3 products, ensure we always output exactly 5 
    # (by filling the rest from the highest overall popular products not already in the list)
    if len(user_recs) < 5:
        for p in popularity_series.index:
            if p not in user_recs:
                user_recs.append(p)
            if len(user_recs) == 5:
                break
                
    final_top5_recommendations.append(user_recs)

print("SUCCESS: Quota 3-2 strategy applied.")
print("\nSample Output for the first 3 customers:")
for i in range(3):
    print(f"Customer {i+1} Recommendations: {final_top5_recommendations[i]}")
print("=====================================================================")