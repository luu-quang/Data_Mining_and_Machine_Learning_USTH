# 🏦 Product Recommendation System for Banking

## 📖 Overview
This project focuses on building a **personalized product recommendation system** for a retail bank.  
The goal is to predict which financial products a customer is most likely to purchase next, based on their historical behavior.

The system helps:
- Improve customer experience
- Increase engagement and satisfaction
- Optimize cross-selling and up-selling strategies

---

## 🎯 Objectives
- Analyze customer behavior and financial activity
- Build a predictive model for product recommendation
- Generate personalized product suggestions for each customer
- Evaluate model performance using **Mean Average Precision (MAP)**

---

## 📊 Dataset
The dataset contains **1.5 years of customer behavior data** from Santander Bank.

- Time range: **Jan 2015 → May 2016**
- Data frequency: Monthly
- Includes:
  - Customer information (age, income, segment, etc.)
  - Product ownership (credit card, savings account, etc.)

### 🧩 Prediction Task
Predict which **new products** a customer will add in:
- 📅 May 2016  
based on:
- 📅 April 2016 data

---

## ⚙️ Project Workflow

### 1. Data Exploration & Cleaning
- Handle missing values
- Remove inconsistencies and outliers
- Perform exploratory data analysis (EDA)

### 2. Feature Engineering
- Create new features from customer behavior
- Select important variables

### 3. Model Training
- Apply machine learning models such as:
  - Logistic Regression
  - Collaborative Filtering
  - Ensemble Models
  - Deep Learning (optional)

### 4. Evaluation
- Metric used: **Mean Average Precision (MAP)**

### 5. Prediction
- Generate ranked product recommendations for each customer

### 6. Optimization
- Improve runtime and scalability
- Optimize code efficiency

---

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost / LightGBM
- Matplotlib / Seaborn

---

## 📈 Results
- Built a recommendation model that predicts customer needs
- Improved targeting of financial products
- Enabled more balanced product exposure across customers

---

## 📂 Project Structure


## Pipeline
[ RAW DATA ENTRANCE ]
                 │
                 ▼
┌─────────────────────────────────┐
│     00_convert_format.py        │  ◄── [STEP 1: INGESTION OPTIMIZATION]
│  • Load Raw CSV (12.7M Rows)     │      • Eliminates I/O bottleneck
│  • Apply Snappy Compression     │      • Slashes load time from 185s to 6s
│  • Convert to Columnar Parquet  │
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     01_eda_and_cleaning.py      │  ◄── [STEP 2: THE SPLIT-BEFORE-RESAMPLE RULE]
│  • train_test_split (80 / 20)   │      • Isolates 2.54M Test rows completely
└─────────────────────────────────┘      • ZERO LOOK-AHEAD BIAS (No Data Leakage)
        │                 │
        ▼ (Train: 10.1M)  ▼ (Test: 2.54M - Isolated)
┌─────────────────────────────────┐
│     01_eda_and_cleaning.py      │  ◄── [STEP 3: ROBUST LÀM SẠCH]
│  • Calculate Train Medians      │      • Outlier Capping on 'renta' via 3-Sigma
│  • Impute Missing 'age' & 'renta'│      • Test set uses strictly TRAIN parameters
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  03_supervised_modeling_...     │  ◄── [STEP 4: TEMPORAL FEATURE MOMENTUM]
│  • Create Lag 1, 2, 3 Months    │      • Captures behavior velocity via .shift()
│  • Calculate Momentum Vector     │      • Converts static rows to time-series paths
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  03_supervised_modeling_...     │  ◄── [STEP 5: DENSE TRAINING SUBSAMPLING]
│  • Sample down to 1.5M Rows     │      • Eliminates redundant background noise
│  • (Test Set remains untouched)  │      • Accelerates training speed by 10x
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  02_feature_extraction_pca.py   │  ◄── [STEP 6: SPATIAL FEATURE FUSION]
│  • StandardScaler (.fit_transform)│      • Mandatory Z-score to balance scales
│  • Extract 19 PCA Components    │      • Retains >85% variance, erases collinearity
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  03_supervised_modeling_...     │  ◄── [STEP 7: PRODUCTION MULTI-LABEL ENGINE]
│  • One-Vs-Rest (OVR) Wrapper    │      • Instantiates 5 parallel tree systems
│  • Balanced Random Forest       │      • Pre-pruned (depth=10, min_split=10)
└─────────────────────────────────┘      • n_jobs=-1 finishes training in 487.83s
                 │
                 ▼
┌─────────────────────────────────┐
│       [ EVALUATION LAYER ]      │  ◄── [STEP 8: PROBABILISTIC RANKING OUTCOME]
│  • Pull predict_proba() continuous arrays (Bypasses hard 50% threshold)
│  • Final Asset Metrics Locked Down:
│    ┌───────────────────────────┬───────────────────────────┐
│    │  MAP@5 Ranking  : 0.7271  │  Global Accuracy: 95.50%  │
│    │  Macro F1-Score : 0.9325  │  Global MAE     : 0.0107  │
│    └───────────────────────────┴───────────────────────────┘
└─────────────────────────────────┘
