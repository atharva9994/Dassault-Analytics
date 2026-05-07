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
    customer_df["_cluster"] = kmeans.fit_predict(X_scaled).astype(str)

# ── Compute per-cluster averages to assign business names ─────────────────────
_summary_agg = {f: "mean" for f in cluster_features}
_summary_agg["Customer_Name"] = "count"
_cs = (
    customer_df.groupby("_cluster")
    .agg(_summary_agg)
    .rename(columns={"Customer_Name": "Customer_Count"})
    .reset_index()
)
for col in cluster_features:
    _cs[col] = _cs[col].round(1)

usage_col = "Avg_Usage_Hours" if "Avg_Usage_Hours" in _cs.columns else "Deal_Count"

# Sequential assignment: most-constrained first
remaining = list(_cs["_cluster"])
name_map: dict = {}

# Small Accounts: lowest Total_Revenue
small_seg = _cs.loc[_cs["Total_Revenue"].idxmin(), "_cluster"]
name_map[small_seg] = "Small Accounts"
remaining.remove(small_seg)

# Champions: highest combined revenue + usage rank among the rest
_rem = _cs[_cs["_cluster"].isin(remaining)].copy()
_rem["_rev_rank"]   = _rem["Total_Revenue"].rank()
_rem["_usage_rank"] = _rem[usage_col].rank()
_rem["_score"]      = _rem["_rev_rank"] + _rem["_usage_rank"]
champ_seg = _rem.loc[_rem["_score"].idxmax(), "_cluster"]
name_map[champ_seg] = "Champions"
remaining.remove(champ_seg)

# At Risk: lowest usage among the two high-revenue survivors
_rem2 = _cs[_cs["_cluster"].isin(remaining)].copy()
at_risk_seg = _rem2.loc[_rem2[usage_col].idxmin(), "_cluster"]
name_map[at_risk_seg] = "At Risk"
remaining.remove(at_risk_seg)

# Growth Accounts: the last remaining cluster
name_map[remaining[0]] = "Growth Accounts"

customer_df["Segment"] = customer_df["_cluster"].map(name_map)

# ── Segment metadata ──────────────────────────────────────────────────────────
SEGMENT_ORDER = ["Champions", "Growth Accounts", "At Risk", "Small Accounts"]

SEGMENT_COLORS = {
    "Champions":       "#2ecc71",
    "Growth Accounts": "#3498db",
    "At Risk":         "#e74c3c",
    "Small Accounts":  "#95a5a6",
}
SEGMENT_DESCRIPTIONS = {
    "Champions":       "High value, high engagement — protect and reward",
    "Growth Accounts": "Growing usage — invest and upsell",
    "At Risk":         "High spend but low usage — intervene now",
    "Small Accounts":  "Low engagement — maintain with minimal effort",
}
SEGMENT_ACTIONS = {
    "Champions":       "Offer loyalty programs, executive briefings, and early access to new features.",
    "Growth Accounts": "Assign dedicated CSMs, schedule QBRs, and present upsell opportunities.",
    "At Risk":         "Trigger immediate outreach, run health checks, and offer re-onboarding.",
    "Small Accounts":  "Automate touchpoints, provide self-serve resources, monitor for churn signals.",
}

# ── Rebuild cluster summary with business names ───────────────────────────────
cluster_summary = (
    customer_df.groupby("Segment")
    .agg({**{f: "mean" for f in cluster_features}, "Customer_Name": "count"})
    .rename(columns={"Customer_Name": "Customer_Count"})
    .reset_index()
)
for col in cluster_features:
    cluster_summary[col] = cluster_summary[col].round(1)

total_rev_by_seg = customer_df.groupby("Segment")["Total_Revenue"].sum()

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.subheader("Segment Overview")

seg_cols = st.columns(4)
for i, seg_name in enumerate(SEGMENT_ORDER):
    row = cluster_summary[cluster_summary["Segment"] == seg_name]
    if row.empty:
        continue
    row = row.iloc[0]
    total_rev = total_rev_by_seg.get(seg_name, 0)
    count = int(row["Customer_Count"])
    color = SEGMENT_COLORS[seg_name]
    desc  = SEGMENT_DESCRIPTIONS[seg_name]

    with seg_cols[i]:
        st.markdown(f"""
<div style="border-left:4px solid {color};padding:10px 14px;background:#fafafa;border-radius:6px;">
  <div style="font-size:1.05rem;font-weight:700;color:{color};">{seg_name}</div>
  <div style="font-size:0.76rem;color:#555;margin-bottom:8px;">{desc}</div>
  <div style="font-size:1.45rem;font-weight:700;">${total_rev:,.0f}</div>
  <div style="font-size:0.82rem;color:#888;">{count} customers</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Scatter plot ──────────────────────────────────────────────────────────────
st.subheader("Revenue vs Usage by Segment")

x_col    = "Total_Revenue"
y_col    = "Avg_Usage_Hours" if "Avg_Usage_Hours" in customer_df.columns else "Deal_Count"
size_col = "Total_Seats"     if "Total_Seats"     in customer_df.columns else None

selected_segments = st.multiselect(
    "Filter by segment",
    options=SEGMENT_ORDER,
    default=SEGMENT_ORDER,
    key="seg_filter",
)
plot_df = customer_df[customer_df["Segment"].isin(selected_segments)]

fig = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    color="Segment",
    size=size_col,
    hover_name="Customer_Name",
    labels={x_col: "Total Revenue (USD)", y_col: y_col.replace("_", " ")},
    color_discrete_map=SEGMENT_COLORS,
    category_orders={"Segment": SEGMENT_ORDER},
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
_order_map = {s: i for i, s in enumerate(SEGMENT_ORDER)}
display_summary["_order"] = display_summary["Segment"].map(_order_map)
display_summary = display_summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

display_summary["Total_Revenue"] = display_summary["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
if "Total_Seats" in display_summary.columns:
    display_summary["Total_Seats"] = display_summary["Total_Seats"].apply(lambda x: f"{x:,.0f}")
if "Avg_Usage_Hours" in display_summary.columns:
    display_summary["Avg_Usage_Hours"] = display_summary["Avg_Usage_Hours"].apply(lambda x: f"{x:.1f} hrs")
display_summary["Deal_Count"] = display_summary["Deal_Count"].apply(lambda x: f"{x:.1f}")
display_summary = display_summary.rename(columns={"Customer_Count": "Customers"})

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

st.divider()

# ── What this means ───────────────────────────────────────────────────────────
st.subheader("What This Means")

for seg_name in SEGMENT_ORDER:
    row = cluster_summary[cluster_summary["Segment"] == seg_name]
    if row.empty:
        continue
    row       = row.iloc[0]
    total_rev = total_rev_by_seg.get(seg_name, 0)
    count     = int(row["Customer_Count"])
    color     = SEGMENT_COLORS[seg_name]
    desc      = SEGMENT_DESCRIPTIONS[seg_name]
    action    = SEGMENT_ACTIONS[seg_name]

    st.markdown(f"""
<div style="border-left:4px solid {color};padding:12px 16px;margin-bottom:12px;background:#fafafa;border-radius:6px;">
  <div style="font-size:1.05rem;font-weight:700;color:{color};">
    {seg_name}&nbsp;<span style="font-weight:400;font-size:0.85rem;color:#555;">— {desc}</span>
  </div>
  <div style="margin-top:6px;display:flex;gap:32px;">
    <span><strong>{count}</strong> customers</span>
    <span><strong>${total_rev:,.0f}</strong> total revenue</span>
  </div>
  <div style="margin-top:6px;font-size:0.88rem;color:#444;"><strong>Recommended action:</strong> {action}</div>
</div>""", unsafe_allow_html=True)

# ── Sidebar chat ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar
render_chat_sidebar()
