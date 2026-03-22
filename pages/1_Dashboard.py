import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Sales Dashboard")

if "data" not in st.session_state:
    st.warning("No data loaded. Please upload a file on the Home page.")
    st.stop()

df = st.session_state["data"].copy()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

def sidebar_multiselect(label, col):
    if col in df.columns:
        options = sorted(df[col].dropna().unique().tolist())
        return st.sidebar.multiselect(label, options, default=options)
    return None

fy_sel      = sidebar_multiselect("Fiscal Year",   "Fiscal_Year")
region_sel  = sidebar_multiselect("Region",        "Region")
product_sel = sidebar_multiselect("Product",       "Product")
lt_sel      = sidebar_multiselect("License Type",  "License_Type")

mask = pd.Series([True] * len(df), index=df.index)
for col, sel in [
    ("Fiscal_Year",  fy_sel),
    ("Region",       region_sel),
    ("Product",      product_sel),
    ("License_Type", lt_sel),
]:
    if sel is not None:
        mask &= df[col].isin(sel)

df = df[mask]

# ── KPI cards ────────────────────────────────────────────────────────────────
st.subheader("Key Metrics")
k1, k2, k3, k4 = st.columns(4)

total_revenue = df["Deal_Value_USD"].sum() if "Deal_Value_USD" in df.columns else 0
deal_count    = len(df)
total_seats   = int(df["Seats"].sum()) if "Seats" in df.columns else 0
high_churn    = int((df["Churn_Risk"] == "High").sum()) if "Churn_Risk" in df.columns else 0

k1.metric("Total Revenue",      f"${total_revenue:,.0f}")
k2.metric("Deal Count",         f"{deal_count:,}")
k3.metric("Total Seats",        f"{total_seats:,}")
k4.metric("High Churn Risk",    f"{high_churn:,}")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
left, right = st.columns(2)

# Revenue by Product
if "Product" in df.columns and "Deal_Value_USD" in df.columns:
    with left:
        prod_rev = (
            df.groupby("Product", as_index=False)["Deal_Value_USD"]
            .sum()
            .sort_values("Deal_Value_USD", ascending=False)
        )
        fig = px.bar(
            prod_rev, x="Product", y="Deal_Value_USD",
            title="Revenue by Product",
            labels={"Deal_Value_USD": "Revenue (USD)"},
            color="Product",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Revenue by Quarter
if "Quarter" in df.columns and "Deal_Value_USD" in df.columns:
    with right:
        qtr_rev = (
            df.groupby("Quarter", as_index=False)["Deal_Value_USD"]
            .sum()
            .sort_values("Quarter")
        )
        fig = px.line(
            qtr_rev, x="Quarter", y="Deal_Value_USD",
            title="Revenue by Quarter",
            labels={"Deal_Value_USD": "Revenue (USD)"},
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

left2, right2 = st.columns(2)

# Revenue by Region
if "Region" in df.columns and "Deal_Value_USD" in df.columns:
    with left2:
        reg_rev = (
            df.groupby("Region", as_index=False)["Deal_Value_USD"]
            .sum()
            .sort_values("Deal_Value_USD", ascending=False)
        )
        fig = px.bar(
            reg_rev, x="Region", y="Deal_Value_USD",
            title="Revenue by Region",
            labels={"Deal_Value_USD": "Revenue (USD)"},
            color="Region",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# License type distribution
if "License_Type" in df.columns:
    with right2:
        lt_counts = df["License_Type"].value_counts().reset_index()
        lt_counts.columns = ["License_Type", "Count"]
        fig = px.pie(
            lt_counts, names="License_Type", values="Count",
            title="License Type Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

# Revenue by Industry
if "Industry" in df.columns and "Deal_Value_USD" in df.columns:
    ind_rev = (
        df.groupby("Industry", as_index=False)["Deal_Value_USD"]
        .sum()
        .sort_values("Deal_Value_USD", ascending=False)
    )
    fig = px.bar(
        ind_rev, x="Industry", y="Deal_Value_USD",
        title="Revenue by Industry",
        labels={"Deal_Value_USD": "Revenue (USD)"},
        color="Industry",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Sidebar chat ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar
render_chat_sidebar()
