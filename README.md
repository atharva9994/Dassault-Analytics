# Dassault Analytics

A Streamlit analytics app for Dassault Systèmes license sales data — built to explore revenue, churn risk, customer segments, and ask plain-English questions about the data.

## Pages

| Page | Description |
|---|---|
| **Home** | Upload a CSV/Excel file or load the built-in sample dataset |
| **Dashboard** | Revenue trends, deal performance, regional breakdown, and KPIs |
| **Churn** | Random Forest churn prediction with feature importance and high-risk customer list |
| **Segments** | KMeans customer segmentation by revenue, deal activity, and usage |
| **Ask Data** | Chat interface — ask business questions in plain English, get computed results |

## Getting Started

### Run locally

```bash
pip install -r requirements.txt
streamlit run App.py
```

### Use the sample dataset

On the Home page, check **"Use sample dataset"** to load the included dataset instantly without uploading a file.

### Bring your own data

Upload a CSV or Excel file with columns such as:

`Customer_Name`, `Deal_Value_USD`, `Deal_Stage`, `Product`, `Region`, `Customer_Segment`, `License_Type`, `Seats`, `Usage_Hours_Per_Month`, `Churn_Risk`, `Booking_Date`

The app auto-cleans the file on upload (deduplication, type coercion, blank filling).

## Ask Data

The Ask Data page uses the **Claude API** (Anthropic). To enable it:

- Add your API key to `.streamlit/secrets.toml`:
  ```toml
  ANTHROPIC_API_KEY = "sk-ant-..."
  ```
- Or paste it directly in the app when prompted.

Example questions:
- *Who are the top 5 customers by revenue?*
- *Which segment has the highest deal value?*
- *Show churn risk by region*

## Tech Stack

- [Streamlit](https://streamlit.io)
- [Plotly](https://plotly.com/python/)
- [scikit-learn](https://scikit-learn.org) — Random Forest, KMeans
- [Anthropic Claude API](https://www.anthropic.com)
- [Pandas](https://pandas.pydata.org)
