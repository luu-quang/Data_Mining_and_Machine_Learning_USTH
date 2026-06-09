# ==============================================================================
# PIPELINE 03 (FINAL SOTA): FLAT 24-PRODUCT INDEPENDENT ENSEMBLE
# Features: Time-Series Lags, PCA (Multicollinearity), RUS (Class Imbalance)
# Evaluation: MAP@5, Recall@5, and Popularity Bias Check
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

print("="*75)
print("PHASE 1 & 2: SOTA PIPELINE (FLAT 24 PRODUCTS + PCA + RUS)")
print("="*75)
start_time = time.time()

# [1] Define the 24 Micro-Products (No Grouping)
product_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

# [2] Load and Chronologically Sort Dataset
print("--- [1/6] Loading Cleaned Parquet Dataset ---")
train_path = r"C:\ML2_final_project\train_cleaned.parquet" 
df = pd.read_parquet(train_path)
df = df.sort_values(by=['ncodpers', 'fecha_dato'])

# [3] Feature Engineering: Temporal Lags & Delta Targets (Anti-Leakage)
print("--- [2/6] Engineering Temporal Lags & Delta Targets ---")
numeric_cols = ['age', 'renta', 'antiguedad']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

delta_targets = []
lag_features = []

for col in product_cols:
    if col in df.columns:
        # Create Lag feature (Month T-1)
        lag_col = f'{col}_lag_1'
        df[lag_col] = df.groupby('ncodpers')[col].shift(1).fillna(0).astype(np.int8)
        lag_features.append(lag_col)
        
        # Create Delta Target (1 ONLY if: Owned this month AND NOT owned last month)
        added_col = f'{col}_added'
        df[added_col] = ((df[col] == 1) & (df[lag_col] == 0)).astype(np.int8)
        delta_targets.append(added_col)

# Drop rows with missing lags (the first month of every customer)
df.dropna(subset=lag_features, inplace=True)

# Separate Features (X) and Targets (y)
leakage_cols = [c for c in df.columns if c in product_cols or c in delta_targets]
X_raw = df.drop(columns=leakage_cols)
y_raw = df[delta_targets]

# Handle Categorical Variables & Missing Values (-1 mapping)
for col in X_raw.columns:
    if col not in numeric_cols:
        if X_raw[col].dtype.name == 'category' or X_raw[col].dtype == 'object':
            X_raw[col] = X_raw[col].astype('category').cat.codes
        else:
            X_raw[col] = X_raw[col].fillna(0).astype(np.int16)

# [4] STRICT ANTI-LEAKAGE RULE: Split BEFORE PCA and Downsampling
print("--- [3/6] Splitting Train and Test Partitions ---")
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)

# [5] Resolve Multicollinearity with PCA (Only on continuous variables)
print("--- [4/6] Applying PCA Spatial Transformations (95% Variance) ---")
X_train_cont = X_train[numeric_cols].fillna(0)
X_test_cont = X_test[numeric_cols].fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_cont)
X_test_scaled = scaler.transform(X_test_cont)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Recombine PCA with Categorical features
X_train_cat = X_train.drop(columns=numeric_cols).values
X_test_cat = X_test.drop(columns=numeric_cols).values

X_train_final = np.hstack((X_train_pca, X_train_cat))
X_test_final = np.hstack((X_test_pca, X_test_cat))

del X_train, X_test, X_raw; gc.collect()

# [6] Train 24 Independent Models (Micro-Personalization)
print("--- [5/6] Executing Targeted Training Set Subsampling (RUS) ---")
predictions_list = []
valid_target_names = []

for idx, target_col in enumerate(delta_targets):
    y_train_single = y_train[target_col].values
    positives = y_train_single.sum()
    
    # Skip dead products (extremely sparse) to save compute and reduce noise
    if positives < 50:
        predictions_list.append(np.zeros(len(X_test_final)))
        valid_target_names.append(target_col)
        continue
        
    print(f" -> Training Model {idx+1:02d}/24: {target_col} | Positives Retained: {positives}")
    
    # Random Under-Sampling: Retain 100% of Positives (Signal), downsample Negatives (Noise) to 1:1
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train_final, y_train_single)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_resampled, y_resampled)
    
    # Predict on the UNTOUCHED Test Set (Preserving natural real-world distribution)
    preds = rf.predict_proba(X_test_final)[:, 1]
    predictions_list.append(preds)
    valid_target_names.append(target_col)

# Combine probabilities into a matrix
raw_probs = np.column_stack(predictions_list)

# [7] Post-Processing: The Lag Mask (Kaggle Hack)
print("--- [6/6] Executing Production Engine Inference ---")
# Multiply by (1 - lag) to mathematically force probability to 0 if they already own it
lag_matrix = df.loc[y_test.index, lag_features].values
final_probs_adjusted = raw_probs * (1 - lag_matrix)

# Extract Top 5 indices per user
top5_indices = np.argsort(final_probs_adjusted, axis=1)[:, -5:][:, ::-1]
final_top5_recommendations = [[product_cols[idx] for idx in user_row] for user_row in top5_indices]

train_time = time.time() - start_time
print(f"Training Execution Time: {train_time:.2f} seconds ({(train_time)/60:.2f} minutes)")

# ==============================================================================
# PHASE 3: EVALUATION (MAP@5 & RECALL@5)
# ==============================================================================
print("\n--- Calculating Recommendation Metrics on Unseen Validation Set ---")

def apk(actual, predicted, k=5):
    """Computes the Average Precision at K (MAP@5 component)"""
    if len(predicted) > k: predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual: return 0.0
    return score / min(len(actual), k)

def recall_at_k(actual, predicted, k=5):
    """Computes the Recall at K (Measures bias and completeness)"""
    if len(predicted) > k: predicted = predicted[:k]
    if not actual: return 0.0
    num_hits = sum([1 for p in predicted if p in actual])
    return num_hits / len(actual)

# Extract actual products the user bought this month for evaluation
actual_purchases = []
y_test_matrix = y_test.values

for i in range(len(y_test_matrix)):
    bought_indices = np.where(y_test_matrix[i] == 1)[0]
    bought_items = [product_cols[idx] for idx in bought_indices]
    actual_purchases.append(bought_items)

# Calculate final MAP and Recall scores
map_scores = []
recall_scores = []

for i in range(len(actual_purchases)):
    if len(actual_purchases[i]) > 0: # Only evaluate users who actually made a purchase
        map_scores.append(apk(actual_purchases[i], final_top5_recommendations[i], k=5))
        recall_scores.append(recall_at_k(actual_purchases[i], final_top5_recommendations[i], k=5))

final_map5 = np.mean(map_scores) if map_scores else 0.0
final_recall5 = np.mean(recall_scores) if recall_scores else 0.0

print("\n" + "="*75)
print(f"FINAL SOTA METRIC - MAP@5 SCORE   : {final_map5:.4f}")
print(f"FINAL SOTA METRIC - RECALL@5 SCORE: {final_recall5:.4f}")
print("="*75)


# ==============================================================================
# PHASE 4: BASELINE CHECK (GLOBAL POPULARITY COVERAGE)
# ==============================================================================
print("\n--- Baseline Check: Coverage of the Top 5 Most Popular Products ---")

# Calculate total new purchases for each product (across the entire y_raw set)
product_counts = y_raw.sum().sort_values(ascending=False)

# Total overall new purchases
total_purchases = product_counts.sum()

# Number of purchases captured by the Top 5 most popular products
top_5_purchases = product_counts.head(5).sum()

# Percentage coverage
coverage = (top_5_purchases / total_purchases) * 100

print(f"Total new purchases (all 24 products) : {total_purchases}")
print(f"Purchases from Top 5 products         : {top_5_purchases}")
print(f"-> Conclusion: The Top 5 popular products account for {coverage:.2f}% of all purchases.")

print("\nDetailed Top 5 Popular Products:")
print(product_counts.head(5))
print("==============================================================================")


# ==============================================================================
# PHASE 5: DISTRIBUTION SHIFT & RARE LABEL DETECTION (POPULARITY BIAS CHECK)
# ==============================================================================
print("\n--- Checking Label Distribution (True vs Predicted in Top 5) ---")

# 1. Gather all actually purchased products
flat_actual = [item for sublist in actual_purchases for item in sublist]

# 2. Gather all predicted products (only for users who actually made a purchase)
flat_predicted = []
for i in range(len(actual_purchases)):
    if len(actual_purchases[i]) > 0:
        flat_predicted.extend(final_top5_recommendations[i])

actual_counts = Counter(flat_actual)
predicted_counts = Counter(flat_predicted)

total_actual = len(flat_actual)
total_predicted = len(flat_predicted)

dist_data = []

for prod in product_cols:
    true_c = actual_counts.get(prod, 0)
    pred_c = predicted_counts.get(prod, 0)
    
    # Calculate Distribution: Quantity of 1 Label / Total Quantity
    true_pct = (true_c / total_actual * 100) if total_actual > 0 else 0
    pred_pct = (pred_c / total_predicted * 100) if total_predicted > 0 else 0
    
    # Calculate Coverage Ratio (Pred_Dist / True_Dist)
    # If ~ 1.0: Model closely matches reality.
    # If > 0 for rare labels: Model successfully retrieves rare labels.
    ratio = (pred_pct / true_pct) if true_pct > 0 else 0
    
    dist_data.append({
        'Product_Name': prod,
        'True_Qty': true_c,
        'Pred_Qty': pred_c,
        'True_Dist(%)': true_pct,
        'Pred_Dist(%)': pred_pct,
        'Coverage_Ratio': ratio
    })

dist_df = pd.DataFrame(dist_data).sort_values(by='True_Qty', ascending=False)

print("\n📊 PRODUCT DISTRIBUTION: ACTUAL vs PREDICTED (TOP 5)")
print("=> True_Dist(%): The actual market share of the product.")
print("=> Pred_Dist(%): The share of the product in the model's recommendations.")
print("=> Coverage_Ratio: Adherence to reality. > 0 for rare products proves NO Popularity Bias!\n")

print(dist_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("==============================================================================")