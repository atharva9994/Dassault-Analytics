import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segments", page_icon="🗂️", layout="wide")
st.title("Customer Segmentation")
st.caption("Customers grouped by revenue, deal activity, and usage patterns using KMeans clustering.")

if "data" not in st.session_state:
    st.warning("No data loaded. Please upload a file on the Home page.")
    st.stop()

df = st.session_state["data"].copy()

if "Customer_Name" not in df.columns:
    st.error("Column 'Customer_Name' not found. Cannot segment customers.")
    st.stop()

# ── Aggregate by customer ─────────────────────────────────────────────────────
agg: dict = {"Deal_Value_USD": "sum", "Deal_ID": "count"}
if "Seats" in df.columns:
    agg["Seats"] = "sum"
if "Usage_Hours_Per_Month" in df.columns:
    agg["Usage_Hours_Per_Month"] = "mean"

customer_df = df.groupby("Customer_Name", as_index=False).agg(agg)
customer_df = customer_df.rename(columns={
    "Deal_Value_USD":        "Total_Revenue",
    "Deal_ID":               "Deal_Count",
    "Seats":                 "Total_Seats",
    "Usage_Hours_Per_Month": "Avg_Usage_Hours",
})

cluster_features = [c for c in ["Total_Revenue", "Total_Seats", "Avg_Usage_Hours", "Deal_Count"]
                    if c in customer_df.columns]

if len(cluster_features) < 2:
    st.error("Not enough numeric features to run clustering.")
    st.stop()

# ── KMeans clustering ─────────────────────────────────────────────────────────
X = customer_df[cluster_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

N_CLUSTERS = 4
with st.spinner("Running segmentation model…"):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    customer_df["Segment"] = kmeans.fit_predict(X_scaled).astype(str)

# ── Cluster summary ────────────────────────────────────────────────────────────
summary_agg = {f: "mean" for f in cluster_features}
summary_agg["Customer_Name"] = "count"
cluster_summary = (
    customer_df.groupby("Segment")
    .agg(summary_agg)
    .rename(columns={"Customer_Name": "Customer_Count"})
    .reset_index()
)
for col in cluster_features:
    cluster_summary[col] = cluster_summary[col].round(1)

# Identify key segments
top_rev_seg    = cluster_summary.loc[cluster_summary["Total_Revenue"].idxmax()]
top_deals_seg  = cluster_summary.loc[cluster_summary["Deal_Count"].idxmax()]
low_rev_seg    = cluster_summary.loc[cluster_summary["Total_Revenue"].idxmin()]

# ── Insights section ──────────────────────────────────────────────────────────
st.subheader("Customer Segmentation Insights")

m1, m2, m3 = st.columns(3)

m1.metric(
    label=f"Highest Revenue — Segment {top_rev_seg['Segment']}",
    value=f"${top_rev_seg['Total_Revenue']:,.0f}",
    help=f"{int(top_rev_seg['Customer_Count'])} customers · avg revenue per customer",
)
m2.metric(
    label=f"Most Active — Segment {top_deals_seg['Segment']}",
    value=f"{int(top_deals_seg['Deal_Count'])} deals avg",
    help=f"{int(top_deals_seg['Customer_Count'])} customers in this segment",
)
m3.metric(
    label=f"Lowest Revenue — Segment {low_rev_seg['Segment']}",
    value=f"${low_rev_seg['Total_Revenue']:,.0f}",
    help=f"{int(low_rev_seg['Customer_Count'])} customers — potential upsell targets",
)

st.divider()

# ── Scatter plot ──────────────────────────────────────────────────────────────
st.subheader("Revenue vs Usage by Segment")

x_col    = "Total_Revenue"
y_col    = "Avg_Usage_Hours" if "Avg_Usage_Hours" in customer_df.columns else "Deal_Count"
size_col = "Total_Seats" if "Total_Seats" in customer_df.columns else None

fig = px.scatter(
    customer_df,
    x=x_col,
    y=y_col,
    color="Segment",
    size=size_col,
    hover_name="Customer_Name",
    labels={x_col: "Total Revenue (USD)", y_col: y_col.replace("_", " ")},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(gridcolor="#f0f0f0"),
    yaxis=dict(gridcolor="#f0f0f0"),
    legend_title="Segment",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Segment summary table ─────────────────────────────────────────────────────
st.subheader("Segment Summary")

display_summary = cluster_summary.copy()
display_summary["Total_Revenue"] = display_summary["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
if "Total_Seats" in display_summary.columns:
    display_summary["Total_Seats"] = display_summary["Total_Seats"].apply(lambda x: f"{x:,.0f}")
if "Avg_Usage_Hours" in display_summary.columns:
    display_summary["Avg_Usage_Hours"] = display_summary["Avg_Usage_Hours"].apply(lambda x: f"{x:.1f} hrs")
display_summary["Deal_Count"] = display_summary["Deal_Count"].apply(lambda x: f"{x:.1f}")

st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.divider()

# ── All customers ─────────────────────────────────────────────────────────────
st.subheader("All Customers")

display_df = customer_df[["Customer_Name", "Segment"] + cluster_features].copy()
display_df = display_df.sort_values("Total_Revenue", ascending=False).reset_index(drop=True)
display_df["Total_Revenue"] = display_df["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
if "Total_Seats" in display_df.columns:
    display_df["Total_Seats"] = display_df["Total_Seats"].apply(lambda x: f"{int(x):,}")
if "Avg_Usage_Hours" in display_df.columns:
    display_df["Avg_Usage_Hours"] = display_df["Avg_Usage_Hours"].apply(lambda x: f"{x:.1f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Sidebar chat ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar
render_chat_sidebar()
