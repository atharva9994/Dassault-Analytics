import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("AI Agent")
st.caption("Ask complex multi-step questions. The agent plans, queries, evaluates, and returns a visual executive report.")

if "data" not in st.session_state:
    st.warning("No data loaded. Please upload a file on the Home page.")
    st.stop()

df = st.session_state["data"]

# ── API key ───────────────────────────────────────────────────────────────────
api_key = (
    st.secrets.get("ANTHROPIC_API_KEY", "")
    or st.session_state.get("anthropic_api_key", "")
)
if not api_key:
    manual = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    if manual:
        st.session_state["anthropic_api_key"] = manual
        api_key = manual
    else:
        st.info("Enter your API key above to enable the AI Agent.")
        st.stop()


# ── LangGraph state schema ────────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    plan: str
    data_results: List[dict]
    sufficient: bool
    metrics: List[dict]
    findings: List[str]
    recommendations: List[str]
    charts: List[dict]          # [{type, step, title}]
    loop_count: int
    error: Optional[str]


# ── JSON extraction helper ────────────────────────────────────────────────────
def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False; continue
        if ch == "\\" and in_str:
            escape = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("Unmatched braces")


# ── Formatting helpers ────────────────────────────────────────────────────────
_COL_LABELS = {
    "Deal_Value_USD": "Revenue ($)",
    "deal_value_usd": "Revenue ($)",
    "Number_of_Seats": "Seats",
    "Seats": "Seats",
    "Usage_Hours_Per_Month": "Usage (hrs/mo)",
    "Churn_Risk": "Churn Risk",
    "Deal_Stage": "Deal Stage",
    "License_Type": "License Type",
    "Customer_Segment": "Segment",
    "Customer_Name": "Customer",
    "Booking_Date": "Booking Date",
}

def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _COL_LABELS.get(c, c) for c in df.columns})

def _fmt_revenue_col(series: pd.Series) -> pd.Series:
    """Format a numeric revenue series as $X.XM or $X,XXX."""
    def _fmt(v):
        try:
            v = float(v)
            if v >= 1_000_000:
                return f"${v / 1_000_000:.1f}M"
            return f"${v:,.0f}"
        except Exception:
            return str(v)
    return series.apply(_fmt)

def _fmt_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and format revenue/dollar columns for display."""
    out = _rename_cols(df.head(20).copy())
    for col in out.columns:
        low = col.lower()
        if any(k in low for k in ["revenue", "value", "amount"]):
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = _fmt_revenue_col(out[col])
    return out


# ── Safe chart helpers ────────────────────────────────────────────────────────
def _best_cat_num(df: pd.DataFrame):
    """Pick the best categorical and numeric columns, excluding Deal_ID."""
    skip = {"deal_id", "id", "index"}
    cat_cols = [c for c in df.columns
                if df[c].dtype == object and c.lower() not in skip]
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in skip]
    return cat_cols, num_cols


def _safe_bar(raw_df: pd.DataFrame, title: str) -> None:
    """Horizontal bar: top 10 by numeric value."""
    try:
        cat_cols, num_cols = _best_cat_num(raw_df)
        if not cat_cols or not num_cols:
            return
        df = raw_df[[cat_cols[0], num_cols[0]]].dropna().nlargest(10, num_cols[0])
        df = _rename_cols(df)
        c, n = df.columns[0], df.columns[1]
        fig = px.bar(df, x=n, y=c, orientation="h", title=title,
                     color=n, color_continuous_scale="Blues", text=n)
        fig.update_layout(coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"},
                          xaxis_showgrid=False, yaxis_showgrid=False,
                          margin={"t": 45, "b": 10, "l": 10, "r": 10},
                          height=max(280, len(df) * 38 + 80), plot_bgcolor="white")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                          marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _safe_pie(raw_df: pd.DataFrame, title: str) -> None:
    """Pie: top 5 + Others."""
    try:
        cat_cols, num_cols = _best_cat_num(raw_df)
        if not cat_cols or not num_cols:
            return
        df = raw_df[[cat_cols[0], num_cols[0]]].dropna().copy()
        df.columns = ["Category", "Value"]
        df = df.groupby("Category", as_index=False)["Value"].sum().sort_values("Value", ascending=False)
        if len(df) > 6:
            others = pd.DataFrame([{"Category": "Others", "Value": df.iloc[5:]["Value"].sum()}])
            df = pd.concat([df.head(5), others], ignore_index=True)
        fig = px.pie(df, names="Category", values="Value", title=title,
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(margin={"t": 45, "b": 10, "l": 10, "r": 10})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _safe_donut(raw_df: pd.DataFrame, title: str) -> None:
    """Donut: same as pie but with hole."""
    try:
        cat_cols, num_cols = _best_cat_num(raw_df)
        if not cat_cols or not num_cols:
            return
        df = raw_df[[cat_cols[0], num_cols[0]]].dropna().copy()
        df.columns = ["Category", "Value"]
        df = df.groupby("Category", as_index=False)["Value"].sum().sort_values("Value", ascending=False)
        if len(df) > 6:
            others = pd.DataFrame([{"Category": "Others", "Value": df.iloc[5:]["Value"].sum()}])
            df = pd.concat([df.head(5), others], ignore_index=True)
        fig = px.pie(df, names="Category", values="Value", title=title,
                     hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(margin={"t": 45, "b": 10, "l": 10, "r": 10})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _safe_line(raw_df: pd.DataFrame, title: str) -> None:
    """Line chart: first date/string col as x, numeric cols as y."""
    try:
        # Prefer a date column, fallback to first categorical
        date_cols = [c for c in raw_df.columns
                     if pd.api.types.is_datetime64_any_dtype(raw_df[c])
                     or any(k in c.lower() for k in ["date", "month", "quarter", "year", "period"])]
        cat_cols, num_cols = _best_cat_num(raw_df)
        x_col = (date_cols or cat_cols or [None])[0]
        if x_col is None or not num_cols:
            return
        df = _rename_cols(raw_df[[x_col] + num_cols[:3]].dropna().copy())
        x_r = df.columns[0]
        y_cols = list(df.columns[1:])
        fig = px.line(df, x=x_r, y=y_cols, title=title, markers=True)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=True,
                          margin={"t": 45, "b": 10, "l": 10, "r": 10},
                          plot_bgcolor="white", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _safe_scatter(raw_df: pd.DataFrame, title: str) -> None:
    """Scatter: first two numeric cols, optional colour from categorical."""
    try:
        cat_cols, num_cols = _best_cat_num(raw_df)
        if len(num_cols) < 2:
            return
        df = _rename_cols(raw_df.copy())
        num_r = [_COL_LABELS.get(c, c) for c in num_cols[:2]]
        color_col = (_COL_LABELS.get(cat_cols[0], cat_cols[0]) if cat_cols else None)
        fig = px.scatter(df, x=num_r[0], y=num_r[1], color=color_col,
                         title=title, size_max=12,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          margin={"t": 45, "b": 10, "l": 10, "r": 10},
                          plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _safe_stacked_bar(raw_df: pd.DataFrame, title: str) -> None:
    """Stacked bar: first cat col as x, remaining cat col as colour, numeric as value."""
    try:
        cat_cols, num_cols = _best_cat_num(raw_df)
        if len(cat_cols) < 2 or not num_cols:
            # Fallback to regular bar if not enough dimensions
            _safe_bar(raw_df, title)
            return
        df = _rename_cols(raw_df[[cat_cols[0], cat_cols[1], num_cols[0]]].dropna().copy())
        c0, c1, n = df.columns[0], df.columns[1], df.columns[2]
        fig = px.bar(df, x=c0, y=n, color=c1, title=title, barmode="stack",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                          margin={"t": 45, "b": 10, "l": 10, "r": 10},
                          plot_bgcolor="white", legend_title=c1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


_CHART_RENDERERS = {
    "bar":          _safe_bar,
    "pie":          _safe_pie,
    "donut":        _safe_donut,
    "line":         _safe_line,
    "scatter":      _safe_scatter,
    "stacked_bar":  _safe_stacked_bar,
    "stacked":      _safe_stacked_bar,
}

def _dispatch_chart(chart_type: str, df: pd.DataFrame, title: str) -> None:
    renderer = _CHART_RENDERERS.get(chart_type.lower(), _safe_bar)
    renderer(df, title)


# ── Metrics renderer ──────────────────────────────────────────────────────────
def _render_metrics(metrics: list) -> None:
    if not metrics:
        return
    for row_start in range(0, len(metrics), 2):
        row = metrics[row_start: row_start + 2]
        cols = st.columns(2)
        for col, m in zip(cols, row):
            label, value = m.get("label", ""), m.get("value", "")
            if len(value) <= 18 and not any(c.isalpha() and c not in "$%KMB" for c in value):
                col.metric(label, value)
            else:
                col.markdown(
                    f"""<div style="background:#f0f2f6;padding:12px 16px;border-radius:8px;margin-bottom:8px;">
<p style="margin:0;font-size:13px;color:#666;">{label}</p>
<p style="margin:0;font-size:16px;font-weight:500;word-break:break-word;">{value}</p>
</div>""",
                    unsafe_allow_html=True,
                )
    st.markdown("")


# ── Main report renderer ──────────────────────────────────────────────────────
def _render_report(final_state: dict, step_dfs: dict) -> None:
    metrics = final_state.get("metrics", [])
    findings = final_state.get("findings", [])
    recommendations = final_state.get("recommendations", [])

    # Guard: nothing to show
    if not metrics and not findings and not recommendations and not step_dfs:
        st.info("No data found for this query. Try a different question.")
        return

    st.markdown("---")

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    st.subheader("Executive Summary")
    _render_metrics(metrics)

    # ── 2. Key Findings ───────────────────────────────────────────────────────
    if findings:
        st.subheader("Key Findings")
        for f in findings:
            st.markdown(f"- {f}")
        st.markdown("")

    # ── 3. Charts ─────────────────────────────────────────────────────────────
    chart_configs = final_state.get("charts", [])[:2]

    # Fallback: if advisor returned no chart config, pick largest df + use bar+pie
    if not chart_configs:
        fallback_df = None
        for result in step_dfs.values():
            if isinstance(result, pd.DataFrame) and not result.empty:
                cat_c, num_c = _best_cat_num(result)
                if cat_c and num_c:
                    if fallback_df is None or len(result) > len(fallback_df):
                        fallback_df = result
        if fallback_df is not None:
            chart_configs = [
                {"type": "bar", "step": None, "title": "Top 10 by Value", "_df": fallback_df},
                {"type": "pie", "step": None, "title": "Distribution",    "_df": fallback_df},
            ]

    if chart_configs:
        st.subheader("Charts")
        cols = st.columns(len(chart_configs))
        for col, cfg in zip(cols, chart_configs):
            df_for_chart = cfg.get("_df")  # pre-resolved fallback
            if df_for_chart is None:
                step_key = cfg.get("step")
                df_for_chart = step_dfs.get(step_key) if step_key else None
                # still none — pick any available df
                if df_for_chart is None:
                    for r in step_dfs.values():
                        if isinstance(r, pd.DataFrame) and not r.empty:
                            df_for_chart = r
                            break
            if df_for_chart is not None:
                with col:
                    _dispatch_chart(cfg.get("type", "bar"), df_for_chart, cfg.get("title", ""))

    # ── 4. Recommendations ────────────────────────────────────────────────────
    if recommendations:
        st.subheader("Recommendations")
        for rec in recommendations:
            st.info(rec)

    # ── 5. Detailed Data (collapsed) ──────────────────────────────────────────
    all_dfs = [r for r in step_dfs.values() if isinstance(r, pd.DataFrame) and not r.empty]
    if all_dfs:
        with st.expander("View detailed data", expanded=False):
            for i, d in enumerate(all_dfs):
                if len(all_dfs) > 1:
                    st.caption(f"Dataset {i + 1}")
                st.dataframe(_fmt_display_df(d), use_container_width=True, hide_index=True)


# ── Graph builder ─────────────────────────────────────────────────────────────
_ANALYST_BASE = """You are a senior business analyst at Dassault Systèmes.
Rules:
- ALWAYS aggregate data by business categories (Customer_Name, Product, Region, Industry, License_Type, Customer_Segment) — NEVER by Deal_ID
- ALWAYS use .nlargest(10) or .head(10) to limit results to at most 10 rows
- Sort results descending by the main numeric column
- Use Deal_Value_USD for revenue; never leave raw floats — round to 2 decimal places
- Return a clean DataFrame with at most 5 columns
- Focus on business insight: totals, averages, counts, rankings
- Return ONLY valid JSON, nothing else"""


def build_graph(api_key: str, df: pd.DataFrame):
    llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=api_key, max_tokens=2048)
    cols = ", ".join(df.columns.tolist())
    sample = df.head(3).to_string(index=False)
    step_dfs: dict = {}

    # Shared execution context — persists variables across all analyst steps
    import numpy as np
    shared_context: dict = {"df": df.copy(), "pd": pd, "np": np}

    # ── Node 1: Planner ───────────────────────────────────────────────────────
    def planner_node(state: AgentState) -> dict:
        try:
            resp = llm.invoke([
                SystemMessage(content=f"""{_ANALYST_BASE}
Dataset columns: {cols}
Sample:
{sample}

Return ONLY a JSON object:
{{"steps": ["Step 1: ...", "Step 2: ..."], "description": "Brief plan summary"}}"""),
                HumanMessage(content=f"Plan the analysis for: {state['question']}"),
            ])
            data = _extract_json(resp.content)
            plan = "\n".join(data.get("steps", [resp.content]))
        except Exception:
            plan = f"Analyse: {state['question']}"
        return {"plan": plan}

    # ── Node 2: Analyst ───────────────────────────────────────────────────────
    def analyst_node(state: AgentState) -> dict:
        prev = "\n".join(
            f"Step {r['step']}: {r['description']}\nResult (first 400 chars): {r['result_str'][:400]}"
            for r in state["data_results"]
        ) or "None yet"

        try:
            resp = llm.invoke([
                SystemMessage(content=f"""{_ANALYST_BASE}
Dataset columns: {cols}
Sample:
{sample}

Return ONLY a JSON object for the NEXT pandas query:
{{"description": "What this query computes", "code": "result = df.groupby('Product')['Deal_Value_USD'].sum().nlargest(10).reset_index()"}}

Critical rules for the code:
- The DataFrame is called `df`. Pandas is `pd`. NumPy is `np`.
- Store the final output in a variable called `result` — do NOT use print().
- result must be a DataFrame (max 10 rows, max 5 columns) OR a scalar.
- You MAY create intermediate variables (e.g. filtered_df, benchmark) — they persist across steps.
- Group by business columns only — never by Deal_ID.
- Always sort descending and limit to top 10.
- Use exact column names from the dataset."""),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Previous results:\n{prev}\n\n"
                    "Generate the next pandas query."
                )),
            ])
            data = _extract_json(resp.content)
            description = data.get("description", "Analysis step")
            code = data.get("code", "")
        except Exception as e:
            return {"data_results": state["data_results"], "loop_count": state["loop_count"] + 1, "error": str(e)}

        result_str = "No result"
        step_num = len(state["data_results"]) + 1

        if code:
            try:
                exec(code, shared_context)
                result = shared_context.get("result")
            except Exception as first_err:
                # Ask Claude to rewrite with a simpler approach
                try:
                    fix_resp = llm.invoke([
                        SystemMessage(content=f"""{_ANALYST_BASE}
The code failed. Rewrite it using only `df` and `pd` with no external variables.
Return ONLY a JSON object: {{"description": "...", "code": "result = ..."}}"""),
                        HumanMessage(content=(
                            f"Failed code:\n{code}\n\n"
                            f"Error: {first_err}\n\n"
                            "Rewrite simpler."
                        )),
                    ])
                    fixed = _extract_json(fix_resp.content)
                    exec(fixed.get("code", ""), shared_context)
                    result = shared_context.get("result")
                    description = fixed.get("description", description)
                except Exception:
                    result = None
                    result_str = "Could not compute this step — continuing with available data."

            if result is not None:
                if isinstance(result, pd.DataFrame):
                    step_dfs[step_num] = result.head(10)
                    result_str = result.head(10).to_string(index=False)
                else:
                    step_dfs[step_num] = result
                    result_str = str(result)

        new_results = state["data_results"] + [{
            "step": step_num,
            "description": description,
            "result_str": result_str,
        }]
        return {"data_results": new_results, "loop_count": state["loop_count"] + 1}

    # ── Node 3: Evaluator ─────────────────────────────────────────────────────
    def evaluator_node(state: AgentState) -> dict:
        summary = "\n".join(
            f"Step {r['step']}: {r['description']}\n{r['result_str'][:300]}"
            for r in state["data_results"]
        )
        try:
            resp = llm.invoke([
                SystemMessage(content='Reply with ONLY one word: "sufficient" or "need_more"'),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Data collected:\n{summary}\n\n"
                    "Is this enough to give specific, data-backed business recommendations?"
                )),
            ])
            sufficient = "sufficient" in resp.content.lower()
        except Exception:
            sufficient = True
        return {"sufficient": sufficient}

    # ── Node 4: Advisor ───────────────────────────────────────────────────────
    def advisor_node(state: AgentState) -> dict:
        full_data = "\n\n".join(
            f"Step {r['step']}: {r['description']}\nResult:\n{r['result_str']}"
            for r in state["data_results"]
        )
        resp = llm.invoke([
            SystemMessage(content=f"""{_ANALYST_BASE}

Return ONLY a JSON object with this exact structure:
{{
  "metrics": [
    {{"label": "Total Revenue", "value": "$101.6M"}},
    {{"label": "High-Risk Accounts", "value": "12"}}
  ],
  "findings": [
    "CATIA accounts for 29% of total revenue ($101.6M)",
    "EMEA region shows highest churn risk with 4 of 12 high-risk accounts"
  ],
  "recommendations": [
    "Prioritize renewal outreach for the 12 high-risk CATIA accounts in EMEA — $2.4M at stake",
    "Investigate low usage in North America subscription accounts before next renewal cycle"
  ],
  "charts": [
    {{"type": "bar", "step": 1, "title": "Revenue by Product"}},
    {{"type": "pie", "step": 1, "title": "Revenue Share by Product"}}
  ]
}}

Chart selection rules — choose the 2 most insightful chart types for the specific question:
- Comparing categories (products, regions, industries, rankings): use "bar"
- Showing composition/share/mix (license type split, segment distribution): use "pie" or "donut"
- Showing trends over time (monthly, quarterly, yearly): use "line"
- Showing correlation (revenue vs usage, seats vs value): use "scatter"
- Showing risk levels by multiple dimensions: use "stacked_bar"
- If question is about trends → line chart, NOT pie
- If question is about risk → stacked_bar or bar, NOT pie
- If question is about share/composition → pie or donut, NOT bar
- Max 2 charts total
- "step" must be the step number from data_results whose DataFrame best fits the chart

Other rules:
- metrics: 2–4 key numbers from the data (format as $X.XM or plain number)
- findings: 3–4 specific data-backed statements with real numbers and percentages
- recommendations: 3–4 short action-oriented bullets (Prioritize/Investigate/Schedule/Consider)
- Every number must come from actual data results, not made up
- RETURN ONLY THE JSON OBJECT. No markdown, no explanation."""),
            HumanMessage(content=f"Question: {state['question']}\n\nData:\n{full_data}"),
        ])
        raw = resp.content.strip()
        try:
            data = _extract_json(raw)
            result = {
                "metrics": data.get("metrics", []),
                "findings": data.get("findings", []),
                "recommendations": data.get("recommendations", []),
                "charts": data.get("charts", []),
            }
            if any(result.values()):
                return result
        except Exception:
            pass
        lines = [l.strip("- •*").strip() for l in raw.splitlines() if len(l.strip()) > 20]
        return {"metrics": [], "findings": [], "recommendations": lines[:5], "charts": []}

    # ── Routing ───────────────────────────────────────────────────────────────
    def route_evaluator(state: AgentState) -> str:
        return "advisor" if (state["sufficient"] or state["loop_count"] >= 3) else "analyst"

    # ── Assemble ──────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("advisor", advisor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "analyst")
    graph.add_edge("analyst", "evaluator")
    graph.add_conditional_edges("evaluator", route_evaluator, {
        "analyst": "analyst",
        "advisor": "advisor",
    })
    graph.add_edge("advisor", END)
    return graph.compile(), step_dfs


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "Which CATIA customers are at churn risk and what should we do?",
    "Compare subscription vs perpetual revenue trends and recommend a strategy",
    "Find underperforming regions and suggest growth opportunities",
    "Identify our top 10 customers and analyze their license patterns",
]

st.markdown("**Example questions:**")
ex_cols = st.columns(len(EXAMPLES))
selected_example = None
for col, example in zip(ex_cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        selected_example = example

question = st.chat_input("Ask a complex business question about your data...")
if selected_example:
    question = selected_example

# ── Session history ───────────────────────────────────────────────────────────
if "agent_history" not in st.session_state:
    st.session_state["agent_history"] = []

for entry in st.session_state["agent_history"]:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        for log in entry.get("steps_log", []):
            with st.expander(log["label"], expanded=False):
                if log.get("df") is not None:
                    st.dataframe(_fmt_display_df(log["df"]), use_container_width=True, hide_index=True)
                elif log.get("text"):
                    st.markdown(log["text"])
        _render_report(entry["final_state"], entry["step_dfs"])

# ── Run agent ─────────────────────────────────────────────────────────────────
NODE_LABELS = {
    "planner":   "Planning analysis...",
    "analyst":   "Querying data...",
    "evaluator": "Evaluating results...",
    "advisor":   "Generating recommendations...",
}

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        steps_log = []
        merged_state = {}

        try:
            graph, step_dfs = build_graph(api_key, df)

            initial_state: AgentState = {
                "question": question,
                "plan": "",
                "data_results": [],
                "sufficient": False,
                "metrics": [],
                "findings": [],
                "recommendations": [],
                "charts": [],
                "loop_count": 0,
                "error": None,
            }
            merged_state = dict(initial_state)

            with st.status("🤔 Starting analysis...", expanded=True) as status:
                for event in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        merged_state.update(node_output)
                        status.update(label=f"⚙️ {NODE_LABELS.get(node_name, node_name)}...")

                        if node_name == "planner" and node_output.get("plan"):
                            with st.expander("📋 Plan", expanded=False):
                                st.markdown(node_output["plan"])
                            steps_log.append({"label": "📋 Plan", "text": node_output["plan"]})

                        elif node_name == "analyst":
                            results = node_output.get("data_results", [])
                            if results:
                                latest = results[-1]
                                step_num = latest["step"]
                                step_df = step_dfs.get(step_num)
                                exp_label = f"📊 Step {step_num}: {latest['description']}"
                                with st.expander(exp_label, expanded=False):
                                    if isinstance(step_df, pd.DataFrame):
                                        st.dataframe(_fmt_display_df(step_df), use_container_width=True, hide_index=True)
                                    elif step_df is not None:
                                        st.metric("Result", str(step_df))
                                    else:
                                        st.markdown(latest["result_str"])
                                steps_log.append({
                                    "label": exp_label,
                                    "df": step_df if isinstance(step_df, pd.DataFrame) else None,
                                    "text": latest["result_str"] if not isinstance(step_df, pd.DataFrame) else None,
                                })

                        elif node_name == "advisor":
                            status.update(label="✅ Analysis complete", state="complete")

        except Exception as e:
            st.error(f"Something went wrong — please try a different question.")

        _render_report(merged_state, step_dfs)

        if merged_state.get("data_results") or merged_state.get("recommendations"):
            st.session_state["agent_history"].append({
                "question": question,
                "steps_log": steps_log,
                "final_state": merged_state,
                "step_dfs": step_dfs,
            })

if st.session_state["agent_history"]:
    if st.button("Clear history"):
        st.session_state["agent_history"] = []
        st.rerun()

render_chat_sidebar()
