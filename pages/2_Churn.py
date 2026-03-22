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
else:
    pass

if "Churn_Risk" not in df.columns:
    st.error("Column 'Churn_Risk' not found. Cannot run churn prediction.")
    st.stop()

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
st.subheader("Feature Importance")
st.caption(
    "Feature importance shows which factors the model found most predictive of churn. "
    "A higher score means that variable had more influence on the prediction — "
    "it does not imply causation, but highlights where to focus retention efforts."
)
importance_df = pd.DataFrame({
    "Feature":   feature_cols,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance_df, x="Importance", y="Feature",
    orientation="h",
    title="Random Forest Feature Importance",
    labels={"Importance": "Importance Score"},
    color="Importance",
    color_continuous_scale="Blues",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

top_feature = importance_df.iloc[0]["Feature"]
st.info(
    f"**Key insight:** `{top_feature}` is the strongest predictor of churn in this dataset. "
    "Customers scoring high on the top features are most likely to churn — "
    "prioritise outreach for high-value accounts in those segments before renewal."
)

st.divider()

# ── High-risk customers ───────────────────────────────────────────────────────
st.subheader("High-Risk Customers")

high_risk_df = df[df["Churn_Predicted"] == 1].copy()

if high_risk_df.empty:
    st.info("No high-risk customers predicted.")
else:
    if "Deal_Value_USD" in high_risk_df.columns:
        high_risk_df = high_risk_df.sort_values("Deal_Value_USD", ascending=False)
        revenue_at_risk = high_risk_df["Deal_Value_USD"].sum()
        st.metric("Total Revenue at Risk", f"${revenue_at_risk:,.0f}")

    display_cols = [c for c in [
        "Customer_Name", "Product", "Region", "Deal_Value_USD",
        "License_Type", "Churn_Risk", "Seats",
    ] if c in high_risk_df.columns]

    st.dataframe(
        high_risk_df[display_cols].reset_index(drop=True),
        use_container_width=True,
    )

# ── Sidebar chat ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar
render_chat_sidebar()
