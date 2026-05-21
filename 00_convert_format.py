# %% [markdown]
# # Phase 1: Data Compression & Memory Management
# ---------------------------------------------------------
"""
TEAMMATE STUDY GUIDE:
Why do we do this? 
Loading a 12.7 million row CSV file takes almost 3 minutes and eats up gigabytes of RAM. 
By converting it to a 'Parquet' file, we compress the data into a columnar format. 
This means our modeling scripts will load the data in 6 seconds instead of 3 minutes, 
preventing our laptops from crashing during the Datathon.
"""

import pandas as pd
import time
import os

print("="*50)
print("PHASE 1: PARQUET COMPRESSION")
print("="*50)

def optimize_and_save(csv_path, parquet_path):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Cannot find {csv_path}")
        return
        
    start_time = time.time()
    print(f"Loading {csv_path}...")
    
    # Read CSV (low_memory=False prevents pandas from guessing data types and crashing)
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Save as Parquet using Snappy compression (Industry Standard)
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"Success! Saved to {parquet_path} in {time.time() - start_time:.2f} seconds.\n")

optimize_and_save(
    r"C:\Users\Laptop K1\Downloads\train.csv", 
    r"C:\Users\Laptop K1\Downloads\train_optimized.parquet"
)
optimize_and_save(
    r"C:\Users\Laptop K1\Downloads\test.csv", 
    r"C:\Users\Laptop K1\Downloads\test_optimized.parquet"
)