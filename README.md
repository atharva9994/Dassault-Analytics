# 3DS License Sales Analytics

A business intelligence and analytics application for Dassault Systèmes license sales data — combining interactive dashboards, machine learning, and LLM-powered data exploration.

## Live App
https://2rtruegbzac9wsf9zwyuay.streamlit.app/

## What's Inside

- **Upload & Clean** — Automated data cleaning pipeline that handles multi-sheet Excel files, removes duplicates, standardizes columns, and fixes missing values
- **Sales Dashboard** — Interactive Plotly charts with KPIs tracking revenue by product (SOLIDWORKS, CATIA, SIMULIA, ENOVIA, DELMIA, 3DEXPERIENCE Platform), region, industry, and license type with real-time filters
- **Churn Prediction** — Random Forest ML model identifying high-risk accounts and quantifying revenue at risk
- **Customer Segmentation** — K-means clustering grouping customers into actionable segments based on revenue, usage, and deal patterns
- **Ask Your Data** — LLM-powered sidebar chat using Claude API for natural language queries on business data
- **AI Agent** — Multi-step reasoning agent that breaks complex questions into steps, runs real Pandas computations, and returns a visual executive report with metric cards, Plotly charts, warnings, and recommendations

## Dataset

Synthetic dataset modeled on Dassault Systèmes' product portfolio with 2,000 license transactions across 6 products, 4 regions, 9 industries, and 4 license types (Subscription, Perpetual, Floating, Node-Locked). Use the "Use sample dataset" checkbox to load it automatically.

## Tech Stack
Python, Pandas, Plotly, Streamlit, scikit-learn, Anthropic Claude API

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
