import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Churn Prediction", page_icon="⚠️", layout="wide")
st.title("Churn Risk Prediction")

if "data" not in st.session_state:
    st.warning("No data loaded. Please upload a file on the Home page.")
    st.stop()

df = st.session_state["data"].copy()

# Filter to Closed Won deals
if "Deal_Stage" in df.columns:
    df = df[df["Deal_Stage"] == "Closed Won"].copy()
    if df.empty:
        st.warning("No 'Closed Won' deals found in the dataset.")
        st.stop()

if "Churn_Risk" not in df.columns:
    st.error("Column 'Churn_Risk' not found. Cannot run churn prediction.")
    st.stop()

# ── How This Works ────────────────────────────────────────────────────────────
with st.expander("How does churn prediction work?"):
    st.markdown("""
- **We trained a Random Forest model** — 100 decision trees each trained on different customer data.
- **Each tree votes** on whether a customer will churn. Majority vote wins.
- **The model was trained on** usage hours, deal value, seats, and license type.
- **It was tested on 20% of customers** it had never seen before to verify accuracy.
""")

st.divider()

# ── Feature encoding ─────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["Product", "License_Type", "Region", "Customer_Segment", "Industry"]
NUMERIC_FEATURES     = ["Deal_Value_USD", "Seats", "Usage_Hours_Per_Month"]

available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
feature_cols  = available_cat + available_num

if not feature_cols:
    st.error("No usable feature columns found for training.")
    st.stop()

X = df[feature_cols].copy()
encoders = {}
for col in available_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

for col in available_num:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

y = (df["Churn_Risk"] == "High").astype(int)

if y.nunique() < 2:
    st.warning("Only one class present in Churn_Risk after filtering. Cannot train classifier.")
    st.stop()

# ── Train model ───────────────────────────────────────────────────────────────
with st.spinner("Training Random Forest model…"):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

df["Churn_Predicted"] = model.predict(X)

# ── Feature importance ────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "Deal_Value_USD":        "Deal Value USD",
    "Usage_Hours_Per_Month": "Monthly Usage Hours",
    "Seats":                 "Number of Seats",
    "License_Type":          "License Type",
    "Product":               "Product",
    "Region":                "Region",
    "Customer_Segment":      "Customer Segment",
    "Industry":              "Industry",
}

importance_df = pd.DataFrame({
    "Feature":    [FEATURE_LABELS.get(f, f) for f in feature_cols],
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

st.subheader("What drives churn risk?")
st.caption(
    "Deal value is the strongest predictor — high-paying customers face more competitive pressure "
    "from alternatives like Siemens NX and PTC Creo."
)

fig = px.bar(
    importance_df, x="Importance", y="Feature",
    orientation="h",
    labels={"Importance": "Importance Score"},
    color="Importance",
    color_continuous_scale="Blues",
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    coloraxis_showscale=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Risk level cards ──────────────────────────────────────────────────────────
st.subheader("Risk Level Overview")

RISK_LEVELS = [
    {
        "label":       "High Risk",
        "filter":      "High",
        "color":       "#e74c3c",
        "icon":        "🔴",
        "timeline":    "Will likely leave within 90 days",
        "action":      "Call immediately, assign senior account manager",
    },
    {
        "label":       "Medium Risk",
        "filter":      "Medium",
        "color":       "#f39c12",
        "icon":        "🟡",
        "timeline":    "Showing early warning signs",
        "action":      "Schedule check-in within 2 weeks",
    },
    {
        "label":       "Low Risk",
        "filter":      "Low",
        "color":       "#2ecc71",
        "icon":        "🟢",
        "timeline":    "Stable accounts",
        "action":      "Continue regular communication",
    },
]

r1, r2, r3 = st.columns(3)
card_cols = [r1, r2, r3]

for col, level in zip(card_cols, RISK_LEVELS):
    subset = df[df["Churn_Risk"] == level["filter"]]
    count  = len(subset)
    rev    = subset["Deal_Value_USD"].sum() if "Deal_Value_USD" in subset.columns else 0
    color  = level["color"]

    with col:
        st.markdown(f"""
<div style="border-left:4px solid {color};padding:12px 16px;background:#fafafa;border-radius:6px;">
  <div style="font-size:1.05rem;font-weight:700;color:{color};">{level['icon']} {level['label']}</div>
  <div style="font-size:1.5rem;font-weight:700;margin-top:6px;">{count} customers</div>
  <div style="font-size:0.9rem;color:#555;">${rev:,.0f} revenue at stake</div>
  <div style="margin-top:8px;font-size:0.82rem;color:#444;">{level['timeline']}</div>
  <div style="margin-top:4px;font-size:0.82rem;color:#666;"><strong>Recommended:</strong> {level['action']}</div>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Top 10 Urgent Accounts ────────────────────────────────────────────────────
st.subheader("Top 10 Urgent Accounts")
st.caption(
    "These accounts represent the highest revenue at risk — prioritize outreach before their renewal dates."
)

high_risk_df = df[df["Churn_Predicted"] == 1].copy()

if high_risk_df.empty:
    st.info("No high-risk customers predicted.")
else:
    urgent_cols = [c for c in [
        "Customer_Name", "Product", "Region",
        "Deal_Value_USD", "Usage_Hours_Per_Month", "Renewal_Date",
    ] if c in high_risk_df.columns]

    urgent_df = (
        high_risk_df[urgent_cols]
        .sort_values("Deal_Value_USD", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    display_urgent = urgent_df.copy()
    if "Deal_Value_USD" in display_urgent.columns:
        display_urgent["Deal_Value_USD"] = display_urgent["Deal_Value_USD"].apply(
            lambda x: f"${x:,.0f}"
        )

    st.dataframe(display_urgent, use_container_width=True, hide_index=True)

st.divider()

# ── Full high-risk list ───────────────────────────────────────────────────────
st.subheader("All High-Risk Customers")

if not high_risk_df.empty:
    display_cols = [c for c in [
        "Customer_Name", "Product", "Region", "Deal_Value_USD",
        "License_Type", "Churn_Risk", "Seats",
    ] if c in high_risk_df.columns]

    full_display = high_risk_df[display_cols].sort_values(
        "Deal_Value_USD", ascending=False
    ).reset_index(drop=True).copy()

    if "Deal_Value_USD" in full_display.columns:
        full_display["Deal_Value_USD"] = full_display["Deal_Value_USD"].apply(
            lambda x: f"${x:,.0f}"
        )

    st.dataframe(full_display, use_container_width=True)

# ── Key Insight ───────────────────────────────────────────────────────────────
st.warning(
    "**Key Insight:** At Risk accounts with low Monthly_Usage_Hours are the most urgent — "
    "customers not using the product have no reason to renew. Focus retention efforts on "
    "high-value accounts with usage below 100 hours/month."
)

# ── Sidebar chat ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar
render_chat_sidebar()
