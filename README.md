# 🩺 Diabetes Risk Prediction Pipeline
### End-to-End Medallion Architecture on Databricks

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Databricks](https://img.shields.io/badge/Databricks-Community-orange?logo=databricks)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.1.0-red?logo=apachespark)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![MLflow](https://img.shields.io/badge/MLflow-3.10.1-blue)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Latest-teal)

---

## 📋 Overview

This project implements a **production-grade, end-to-end data pipeline** for diabetes risk prediction using the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021 dataset.

The pipeline follows the **Medallion Architecture** (Bronze → Silver → Gold) on Databricks, with an XGBoost ML model achieving **AUC = 0.878** — demonstrating that self-reported survey data alone is sufficient for clinically meaningful diabetes risk prediction.

---

## 🏆 Key Results

| Metric | Value | Threshold |
|--------|-------|-----------|
| AUC-ROC | **0.878** | > 0.80 ✅ |
| Recall (diabetic patients caught) | **81.5%** | > 75% ✅ |
| F1 Score | **0.795** | > 0.75 ✅ |
| Precision | **0.777** | > 0.75 ✅ |
| Val vs Test variance | **< 0.01** | < 0.02 ✅ |

---

## 🏗️ Architecture

```
📁 SOURCE FILES (CDC BRFSS 2021)
   3 CSV files → Volume storage
          ↓ Autoloader (File Arrival Trigger)
🥉 BRONZE LAYER
   Raw ingestion — 539,892 rows
   Schema evolution, audit columns
          ↓
🥈 SILVER LAYER
   Clean data — 445,149 rows
   Type casting, dedup, BMI validation
   Feature engineering
          ↓
🥇 GOLD LAYER
   4 tables for different consumers
   ├── gold_diabetes     → ETL/BI (445,149 rows)
   ├── gold_analytics    → Business insights (16 rows)
   ├── gold_features     → ML + SMOTE (379,640 rows)
   └── gold_predictions  → Risk tiers (56,946 rows)
          ↓
🤖 ML PIPELINE
   3 models trained + MLflow tracking
   SHAP explainability
   XGBoost AUC = 0.878
```

---

## 📁 Repository Structure

```
├── 01_Bronze_Ingestion.ipynb        # Autoloader ingestion
├── 02_Silver_Transformation.ipynb   # Cleaning + feature engineering
├── 03_Gold_FeatureStore.ipynb       # Analytics + ML features + SMOTE
├── 04_ML_Training.ipynb             # Model training + MLflow + SHAP
└── README.md                        # This file
```

---

## 📊 Dataset

**Source:** [Kaggle — CDC Diabetes Health Indicators (BRFSS 2021)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| File | Rows | Description |
|------|------|-------------|
| diabetes_binary_health_indicators_BRFSS2021.csv | 236,378 | Real-world class imbalance (binary) |
| diabetes_binary_5050split_health_indicators_BRFSS2021.csv | 303,514 | Balanced binary split |
| diabetes_012_health_indicators_BRFSS2021.csv | 236,378 | 3-class (0=No, 1=Pre, 2=Yes) |

**22 Features including:**
- `Diabetes_binary` — Target (0 = No Diabetes, 1 = Diabetes)
- Numerical: BMI, Age, GenHlth, MentHlth, PhysHlth, Education, Income
- Binary (14): HighBP, HighChol, Smoker, Stroke, PhysActivity, and more
- Engineered: `Health_Risk_Score`, `BMI_Category`, `Age_Group`

---

## 🔧 Delta Lake Features

| Feature | Implementation |
|---------|---------------|
| ACID Transactions | All writes use Delta format |
| Time Travel | 5+ versions per table — full audit trail |
| OPTIMIZE + ZORDER | Applied after every write for faster queries |
| Schema Evolution | `overwriteSchema` + `_rescued_data` column |
| Autoloader Checkpoint | Exactly-once incremental ingestion |

```python
# Time travel example
spark.read.option("versionAsOf", 0).table("diabetes_medallion.silver_diabetes")
```

---

## 🤖 Machine Learning

### Model Comparison

| Model | AUC-ROC | Recall | F1 | Training Time |
|-------|---------|--------|----|---------------|
| Logistic Regression | 0.8141 | 75.4% | 0.7417 | ~4 mins |
| Random Forest | 0.8310 | 79.5% | 0.7602 | ~2 mins |
| **XGBoost** ⭐ | **0.8750** | **81.5%** | **0.7920** | **~15 secs** |

### SHAP Top 5 Risk Factors

| Rank | Feature | SHAP Value | Insight |
|------|---------|------------|---------|
| 1 | Health_Risk_Score (engineered) | 0.6126 | Composite score most predictive |
| 2 | BMI | 0.3657 | Obese = 3.6x higher risk |
| 3 | Smoker | 0.3540 | Ranked above Age — insulin resistance |
| 4 | Age | 0.2728 | Elderly = 9x higher than young adults |
| 5 | GenHlth | 0.2259 | Self-reported health is reliable proxy |

### Risk Tier Predictions

| Tier | Patients | Accuracy | Action |
|------|----------|----------|--------|
| 🔴 High Risk | 24,547 (43%) | 82.8% | Immediate screening |
| 🟡 Medium Risk | 14,757 (26%) | 59.8% | Monitor closely |
| 🟢 Low Risk | 17,642 (31%) | 89.9% | Routine checkup |

---

## ⚙️ Pipeline Automation

The pipeline is **fully automated** using Databricks Jobs:

- **File Arrival Trigger** — dropping a CSV automatically starts the pipeline
- **4-task orchestration** — each notebook waits for the previous to succeed
- **Repair Run** — failed tasks rerun independently without restarting from scratch
- **Idempotent** — safe to rerun multiple times, same results guaranteed
- **Incremental loading** — Autoloader processes only new files via checkpoint

```
Drop new CSV file
      ↓
Databricks detects automatically
      ↓
Full pipeline runs: Bronze → Silver → Gold → ML
      ↓
Zero human intervention required! 🚀
```

---

## 🚀 Setup & Reproduction

### Prerequisites
- Databricks Community Edition account
- Unity Catalog enabled
- Kaggle CDC BRFSS 2021 dataset (3 CSV files)

### Step 1 — Create Storage Volume
```sql
CREATE CATALOG IF NOT EXISTS workspace;
CREATE SCHEMA IF NOT EXISTS workspace.diabetes_pipeline;
CREATE VOLUME IF NOT EXISTS workspace.diabetes_pipeline.healthcare_db;
```

### Step 2 — Upload Dataset
Upload CSV files to:
```
/Volumes/workspace/diabetes_pipeline/healthcare_db/
```

### Step 3 — Run Notebooks in Order
```
01_Bronze_Ingestion        → Ingest raw data
02_Silver_Transformation   → Clean and transform
03_Gold_FeatureStore       → Build analytics + ML features
04_ML_Training             → Train models + SHAP
```

### Step 4 — Configure Automated Job
1. Create Job: `diabetes_pipeline_workflow`
2. Add 4 tasks with dependencies
3. Add **File Arrival Trigger** → point to `healthcare_db` volume
4. Drop any new CSV → pipeline runs automatically!

---

## 📈 Business Impact

| Finding | Impact |
|---------|--------|
| Obese patients 3.6x more likely to have diabetes | Target weight management programs |
| Elderly patients 9x more likely than young adults | Prioritize screening for 55+ |
| Smoking ranked above age as risk factor | Smoking cessation reduces diabetes risk |
| Risk Score 0→7: 0.65%→29.33% diabetes rate | Simple 7-point score for rapid triage |
| Model catches 81.5% of diabetic patients | 6,100 extra patients identified per 100,000 |

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Cloud Platform | Databricks Community Edition | Serverless |
| Processing Engine | Apache Spark | 4.1.0 |
| Storage Format | Delta Lake | Latest |
| ML Tracking | MLflow | 3.10.1 |
| Primary Model | XGBoost | 3.2.0 |
| Explainability | SHAP | 0.51.0 |
| Class Balancing | imbalanced-learn (SMOTE) | 0.14.1 |
| Language | Python / PySpark | 3.12 |

---

## 📝 Data Quality Summary

| Layer | Rows | Notes |
|-------|------|-------|
| Bronze | 539,892 | Raw — all 3 files, schema evolution handled |
| Silver | 445,149 | 94,743 removed (duplicates + BMI outliers) |
| Gold (ETL/BI) | 445,149 | Full real data — no sampling |
| Gold (ML) | 379,640 | SMOTE applied for class balance |
| Predictions | 56,946 | Test set predictions with risk tiers |

---

## 📄 License

This project uses the CDC BRFSS 2021 public dataset available on Kaggle.
Pipeline code is free to use for educational and portfolio purposes.

---

*Built with ❤️ using Databricks, Delta Lake, XGBoost, MLflow and SHAP*# diabetes-risk-prediction-pipeline
