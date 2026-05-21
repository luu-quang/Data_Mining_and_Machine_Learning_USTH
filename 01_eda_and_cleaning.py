# %% [markdown]
# # Phase 2: Synchronized Cleaning & Verification
# ---------------------------------------------------------
"""
TEAMMATE STUDY GUIDE (Refer to Lecture 2 - Data Preparation):
1. Data Leakage Warning: We MUST calculate the median income (renta) from the TRAIN data 
   and use that exact same number to fill missing values in the TEST data. If we calculate 
   a new median for the test data, it is called "Data Leakage" and we will fail the lab.
2. 3-Sigma Rule: Instead of deleting million-dollar incomes, we cap them at 3 Standard 
   Deviations from the mean to remove extreme outliers without losing customer rows.
"""

import pandas as pd
import gc

print("="*60)
print("PHASE 2: SYNCHRONIZED DATA CLEANING")
print("="*60)

def process_data(df, dataset_name, median_renta=None, median_age=None):
    print(f"\n--- Processing {dataset_name} ---")
    nulls_before = df.isnull().sum()
    
    # [Lecture 2] Deduplication
    df = df.drop_duplicates()
    
    # Coerce errors: If someone typed "Thirty" instead of "30", turn it into NaN
    df['renta'] = pd.to_numeric(df['renta'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # [Lecture 2] Synchronized Imputation (Protecting against Data Leakage)
    if median_renta is None: median_renta = df['renta'].median()
    if median_age is None: median_age = df['age'].median()
    
    df['renta'] = df['renta'].fillna(median_renta)
    df['age'] = df['age'].fillna(median_age)
    
    # [Lecture 2] Outlier Capping (3-Sigma Rule for Renta, Logical Bounds for Age)
    df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = int(median_age)
    upper_bound = df['renta'].mean() + (3 * df['renta'].std())
    df.loc[df['renta'] > upper_bound, 'renta'] = upper_bound
    
    # Categorical Imputation: Replace empty demographic strings with an explicit 'UNKNOWN' category
    cat_cols = ['cod_prov', 'canal_entrada', 'pais_residencia', 'segmento', 'sexo', 'ind_empleado']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', 'None', '', ' '], 'UNKNOWN').astype('category')
            
    # Verification Printout for the Lab Report
    nulls_after = df.isnull().sum()
    null_stats = pd.DataFrame({'Before Imputation': nulls_before, 'After Imputation': nulls_after})
    print("\n[Imputation Verification Matrix]:")
    missing_matrix = null_stats[null_stats['Before Imputation'] > 0]
    print(missing_matrix if not missing_matrix.empty else "No missing values.")
    
    return df, median_renta, median_age

# Execute Pipeline
train_df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\train_optimized.parquet")
train_df, m_renta, m_age = process_data(train_df, "TRAINING DATA")

test_df = pd.read_parquet(r"C:\Users\Laptop K1\Downloads\test_optimized.parquet")
test_df, _, _ = process_data(test_df, "TESTING DATA", m_renta, m_age)

train_df.to_parquet(r"C:\Users\Laptop K1\Downloads\train_cleaned.parquet", engine='pyarrow')
test_df.to_parquet(r"C:\Users\Laptop K1\Downloads\test_cleaned.parquet", engine='pyarrow')

del train_df, test_df
gc.collect()