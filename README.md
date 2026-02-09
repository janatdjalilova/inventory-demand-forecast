# Inventory Demand Forecasting (Kaspi Coffee)

## 📌 Project Overview
This project focuses on demand forecasting and inventory optimization for coffee products sold on the Kaspi marketplace.  
Machine learning models are used to predict product popularity and expected demand, which are then translated into inventory metrics such as safety stock and reorder point.

## 🎯 Business Problem
Retailers on marketplaces face two major risks:
- stockouts → lost sales and lower ratings
- overstock → higher holding costs and frozen capital

The goal of this project is to support inventory planning decisions using data-driven demand forecasts.

## 📊 Data
- Source: Kaspi marketplace (coffee category)
- Data includes:
  - product price
  - rating and reviews
  - brand information
  - popularity indicators
- Target variables:
  - product popularity (classification)
  - expected number of reviews (regression, proxy for demand)

## 🧠 Modeling Approach
Two ML models are used:
- **Classification model** — predicts probability that a product is popular
- **Regression model** — predicts expected demand (number of reviews)

Based on predicted demand, classical inventory metrics are calculated.

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- RandomForest
- Matplotlib
- Streamlit (interactive dashboard)

## ⚙️ ML & Inventory Pipeline
1. Data cleaning and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature engineering  
4. Model training (classification + regression)  
5. Threshold tuning (F1-score)  
6. Demand estimation (proxy via predicted reviews)  
7. Inventory calculations (safety stock, reorder point, turnover)  
8. Streamlit dashboard (SKU + portfolio views)

## 📈 Inventory Metrics Used
- **Daily demand**
- **Safety stock**
- **Reorder point**
- **Annual demand**
- **Inventory turnover**

All metrics are configurable using business assumptions such as lead time, service level and demand variability.

## 📊 Application Features
- Single SKU analysis
- Portfolio-level dashboard
- Brand, price and rating filters
- Downloadable inventory table
- Feature importance visualization

## 📂 Project Structure
```text
inventory-demand-forecast/
├─ app/
│  └─ app.py
├─ notebooks/
│  └─ Kaspi_Inventory_Project.ipynb
├─ data/
│  ├─ kaspi_coffee_raw.csv
│  └─ kaspi_coffee_cleaned.csv
├─ models/
│  ├─ model_popularity_rf.pkl
│  └─ model_reviews_rf.pkl
├─ requirements.txt
└─ README.md
