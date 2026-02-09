# Inventory Demand Forecasting (Kaspi Coffee)

## 📌 Project Overview
This project focuses on demand forecasting and inventory optimization for coffee products sold on Kaspi marketplace.  
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
2. Feature engineering
3
