# Automated ML Model Monitoring & Drift Detection Pipeline

## Overview
This repository contains an end-to-end, automated machine learning model monitoring pipeline designed for quantitative finance applications. It tracks the degradation of a credit risk model (Random Forest) in a simulated production environment over a 30-day period.

## Technical Architecture
* **Language:** Python (Pandas, Scikit-Learn, SciPy)
* **Storage/Logging:** SQLite
* **Visualization:** Tableau
* **Domain:** Quantitative Finance / Credit Risk

## Core Pipeline Components
1. **The "Time Machine" (`prepare_data.py`):** Ingests historical financial data and chronologically splits it to simulate 30 days of future production data.
2. **Baseline Risk Model (`train_baseline.py`):** A Random Forest Classifier trained to predict binary loan defaults.
3. **The Automated Drift Engine (`daily_monitor.py`):** Iteratively loads daily production batches, generates baseline predictions, and executes manual Kolmogorov-Smirnov (KS) tests to calculate statistical distribution shifts across critical macroeconomic variables.
4. **Alerting & Logging:** Autonomously determines the model's status and appends results to a local SQLite database.

## 📊 Data Source
The model was developed and tested using the **Lending Club Loan Data** (1.3GB, ~2.2M records). 
Due to GitHub file size constraints, the raw dataset is not hosted here. 
You can access the source data on Kaggle: [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
