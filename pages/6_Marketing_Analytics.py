import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Marketing Analytics", layout="wide")
st.title("Marketing Analytics")

if "data" not in st.session_state:
    st.info("Upload your data on the Home page to get started.")
    st.stop()

df_full = st.session_state["data"]

required_cols = {"Campaign_Source", "Marketing_Channel", "Impressions", "Clicks",
                 "Leads_Generated", "Campaign_Cost", "Web_Visits",
                 "Email_Open_Rate", "Email_Click_Rate"}

if not required_cols.issubset(df_full.columns):
    st.error("Marketing columns not found in this dataset. Please upload the updated sample dataset that includes campaign data.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

campaign_sources = sorted(df_full["Campaign_Source"].dropna().unique())
selected_sources = st.sidebar.multiselect(
    "Campaign Source",
    campaign_sources,
    default=campaign_sources,
)

channels = sorted(df_full["Marketing_Channel"].dropna().unique())
selected_channels = st.sidebar.multiselect(
    "Marketing Channel",
    channels,
    default=channels,
)

fiscal_years = sorted(df_full["Fiscal_Year"].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Fiscal Year",
    fiscal_years,
    default=fiscal_years,
)

df = df_full[
    df_full["Campaign_Source"].isin(selected_sources) &
    df_full["Marketing_Channel"].isin(selected_channels) &
    df_full["Fiscal_Year"].isin(selected_years)
].copy()

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ── Derived metrics ───────────────────────────────────────────────────────────
total_leads = int(df["Leads_Generated"].sum())
total_conversions = int((df["Deal_Status"] == "Closed Won").sum())
total_campaign_cost = df["Campaign_Cost"].sum()
total_revenue = df.loc[df["Deal_Status"] == "Closed Won", "Deal_Value_USD"].sum()
avg_cpl = total_campaign_cost / total_leads if total_leads > 0 else 0
roi = ((total_revenue - total_campaign_cost) / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Leads Generated", f"{total_leads:,}")
k2.metric("Total Conversions", f"{total_conversions:,}")
k3.metric("Avg Cost Per Lead", f"${avg_cpl:,.0f}")
k4.metric("Overall Campaign ROI", f"{roi:.1f}%")

st.divider()

# ── Row 1: Funnel + Channel Performance ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Marketing Funnel")
    funnel_data = {
        "Stage": ["Impressions", "Clicks", "Leads", "Conversions"],
        "Count": [
            int(df["Impressions"].sum()),
            int(df["Clicks"].sum()),
            total_leads,
            total_conversions,
        ],
    }
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["Stage"],
        x=funnel_data["Count"],
        textinfo="value+percent initial",
        marker=dict(color=["#1f77b4", "#4a9ede", "#7dbfe8", "#aad4f0"]),
    ))
    fig_funnel.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380)
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.caption("End-to-end conversion funnel showing drop-off from impressions to closed deals.")

with col2:
    st.subheader("Performance by Marketing Channel")
    channel_df = (
        df.groupby("Marketing_Channel")
        .agg(
            Leads=("Leads_Generated", "sum"),
            Cost=("Campaign_Cost", "sum"),
        )
        .reset_index()
    )
    channel_df["Cost_Per_Lead"] = (channel_df["Cost"] / channel_df["Leads"]).round(0)

    fig_channel = px.bar(
        channel_df.sort_values("Leads", ascending=False),
        x="Marketing_Channel",
        y=["Leads", "Cost_Per_Lead"],
        barmode="group",
        labels={"value": "Value", "variable": "Metric", "Marketing_Channel": "Channel"},
        color_discrete_map={"Leads": "#1f77b4", "Cost_Per_Lead": "#ff7f0e"},
    )
    fig_channel.update_layout(margin=dict(t=20, b=20), height=380, legend_title="")
    st.plotly_chart(fig_channel, use_container_width=True)
    st.caption("Grouped bars compare lead volume against cost efficiency per channel. Lower cost-per-lead signals better ROI.")

st.divider()

# ── Row 2: Campaign ROI + Web Traffic Trend ───────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Campaign ROI by Source")
    roi_df = (
        df.groupby("Campaign_Source")
        .apply(lambda g: pd.Series({
            "Revenue": g.loc[g["Deal_Status"] == "Closed Won", "Deal_Value_USD"].sum(),
            "Cost": g["Campaign_Cost"].sum(),
        }))
        .reset_index()
    )
    roi_df["ROI_Pct"] = ((roi_df["Revenue"] - roi_df["Cost"]) / roi_df["Cost"] * 100).round(1)

    fig_roi = px.bar(
        roi_df.sort_values("ROI_Pct", ascending=True),
        x="ROI_Pct",
        y="Campaign_Source",
        orientation="h",
        labels={"ROI_Pct": "ROI (%)", "Campaign_Source": "Campaign"},
        color="ROI_Pct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig_roi.update_layout(margin=dict(t=20, b=20), height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_roi, use_container_width=True)
    st.caption("ROI = (Revenue − Cost) / Cost × 100. Green bars indicate campaigns returning more than their spend.")

with col4:
    st.subheader("Web Traffic Trend")
    if "Booking_Date" in df.columns:
        trend_df = df.copy()
        trend_df["Month"] = pd.to_datetime(trend_df["Booking_Date"]).dt.to_period("M").dt.to_timestamp()
        trend_df = (
            trend_df.groupby("Month")["Web_Visits"]
            .sum()
            .reset_index()
            .sort_values("Month")
        )
        fig_trend = px.line(
            trend_df,
            x="Month",
            y="Web_Visits",
            labels={"Month": "Month", "Web_Visits": "Web Visits"},
            markers=True,
        )
        fig_trend.update_traces(line_color="#1f77b4")
        fig_trend.update_layout(margin=dict(t=20, b=20), height=380)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Monthly web visit volume driven by all campaigns combined. Peaks often correlate with product launches or trade shows.")
    else:
        st.info("Booking_Date column required for time-series trend.")

st.divider()

# ── Email Metrics Table ───────────────────────────────────────────────────────
st.subheader("Email Engagement by Campaign Source")
email_df = (
    df.groupby("Campaign_Source")
    .agg(
        Avg_Open_Rate=("Email_Open_Rate", "mean"),
        Avg_Click_Rate=("Email_Click_Rate", "mean"),
        Total_Leads=("Leads_Generated", "sum"),
    )
    .round(2)
    .reset_index()
    .sort_values("Avg_Open_Rate", ascending=False)
    .rename(columns={
        "Campaign_Source": "Campaign Source",
        "Avg_Open_Rate": "Avg Open Rate (%)",
        "Avg_Click_Rate": "Avg Click Rate (%)",
        "Total_Leads": "Total Leads",
    })
)
st.dataframe(email_df, use_container_width=True, hide_index=True)
st.caption("Email open and click rates by campaign type. Higher click rates indicate more engaged audiences.")
