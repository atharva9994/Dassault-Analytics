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
    data_results: List[dict]   # [{step, description, result_str}]
    sufficient: bool
    metrics: List[dict]        # [{label, value}]
    recommendations: List[str]
    loop_count: int
    error: Optional[str]


# ── JSON extraction helper ────────────────────────────────────────────────────
def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    # Bracket-counter: find the first complete {...}
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


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph(api_key: str, df: pd.DataFrame):
    """
    Returns (compiled_graph, step_dfs) where step_dfs is a shared dict
    that analyst_node populates with DataFrames keyed by step number.
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=api_key, max_tokens=2048)
    cols = ", ".join(df.columns.tolist())
    sample = df.head(3).to_string(index=False)
    step_dfs: dict = {}

    # ── Node 1: Planner ───────────────────────────────────────────────────────
    def planner_node(state: AgentState) -> dict:
        try:
            resp = llm.invoke([
                SystemMessage(content=f"""You are a data analyst for Dassault Systèmes.
Dataset columns: {cols}
Sample:
{sample}

Return ONLY a JSON object describing the analysis plan:
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
                SystemMessage(content=f"""You are a Python/Pandas analyst.
Dataset columns: {cols}
Sample:
{sample}

Return ONLY a JSON object for the NEXT pandas query needed:
{{"description": "What this query computes", "code": "result = df.groupby('Product')['Deal_Value_USD'].sum().reset_index()"}}

Rules:
- `code` must store output in a variable named `result`
- result must be a DataFrame (max 10 rows, max 5 columns) or a scalar
- Use exact column names from the dataset
- Return ONLY valid JSON, nothing else"""),
                HumanMessage(content=(
                    f"Question: {state['question']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Previous results:\n{prev}\n\n"
                    "Generate the next pandas query needed to answer the question."
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
                local_vars = {"df": df.copy(), "pd": pd}
                exec(code, {}, local_vars)
                result = local_vars.get("result")
                if isinstance(result, pd.DataFrame):
                    step_dfs[step_num] = result
                    result_str = result.head(10).to_string(index=False)
                elif result is not None:
                    step_dfs[step_num] = result
                    result_str = str(result)
            except Exception as e:
                result_str = f"Execution error: {e}"

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
                    f"Data collected so far:\n{summary}\n\n"
                    "Is this enough to give specific, data-backed business recommendations?"
                )),
            ])
            sufficient = "sufficient" in resp.content.lower()
        except Exception:
            sufficient = True  # on error, proceed to advisor
        return {"sufficient": sufficient}

    # ── Node 4: Advisor ───────────────────────────────────────────────────────
    def advisor_node(state: AgentState) -> dict:
        full_data = "\n\n".join(
            f"Step {r['step']}: {r['description']}\nResult:\n{r['result_str']}"
            for r in state["data_results"]
        )
        resp = llm.invoke([
            SystemMessage(content=f"""You are a senior business advisor for Dassault Systèmes.
Return ONLY a JSON object with this exact structure:
{{
  "metrics": [
    {{"label": "Total Revenue at Risk", "value": "$2,450,000"}},
    {{"label": "High-Risk Customers", "value": "12"}}
  ],
  "recommendations": [
    "Prioritise CATIA renewal outreach in EMEA — 6 accounts, $1.2M at stake",
    "Offer usage training to accounts with low engagement before renewal"
  ]
}}

Rules:
- metrics: 2–4 key numbers pulled from the actual data results above
- recommendations: 2–4 short actionable bullets with specific numbers from the data
- RETURN ONLY THE JSON OBJECT. No explanation, no markdown, no extra text."""),
            HumanMessage(content=f"Question: {state['question']}\n\nData collected:\n{full_data}"),
        ])
        raw = resp.content.strip()
        # Try structured parse first
        try:
            data = _extract_json(raw)
            metrics = data.get("metrics", [])
            recommendations = data.get("recommendations", [])
            if metrics or recommendations:
                return {"metrics": metrics, "recommendations": recommendations}
        except Exception:
            pass
        # Fallback: extract any bullet lines as recommendations, show raw as one rec
        lines = [l.strip("- •*").strip() for l in raw.splitlines() if l.strip()]
        recs = [l for l in lines if len(l) > 20][:5]
        return {"metrics": [], "recommendations": recs if recs else [raw[:500]]}

    # ── Conditional routing ───────────────────────────────────────────────────
    def route_evaluator(state: AgentState) -> str:
        if state["sufficient"] or state["loop_count"] >= 3:
            return "advisor"
        return "analyst"

    # ── Assemble graph ────────────────────────────────────────────────────────
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


# ── Visual helpers ────────────────────────────────────────────────────────────
def _auto_bar(result_df: pd.DataFrame, title: str) -> None:
    cat_cols = [c for c in result_df.columns if result_df[c].dtype == object]
    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    if not cat_cols or not num_cols:
        return
    fig = px.bar(result_df, x=num_cols[0], y=cat_cols[0], orientation="h",
                 title=title, color=num_cols[0], color_continuous_scale="Blues",
                 text=num_cols[0])
    fig.update_layout(coloraxis_showscale=False,
                      yaxis={"categoryorder": "total ascending"},
                      margin={"t": 40, "b": 20},
                      height=max(300, len(result_df) * 40 + 80))
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def _auto_pie(result_df: pd.DataFrame, title: str) -> None:
    cat_cols = [c for c in result_df.columns if result_df[c].dtype == object]
    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    if not cat_cols or not num_cols:
        return
    fig = px.pie(result_df, names=cat_cols[0], values=num_cols[0], title=title,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(margin={"t": 40, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


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


def _render_report(final_state: dict, step_dfs: dict) -> None:
    st.markdown("---")
    st.markdown("### Executive Summary")

    _render_metrics(final_state.get("metrics", []))

    # Charts — use largest DataFrame available
    chart_df = None
    for result in step_dfs.values():
        if isinstance(result, pd.DataFrame) and not result.empty:
            if chart_df is None or len(result) > len(chart_df):
                chart_df = result

    if chart_df is not None:
        cat_count = sum(1 for c in chart_df.columns if chart_df[c].dtype == object)
        num_count = sum(1 for c in chart_df.columns if pd.api.types.is_numeric_dtype(chart_df[c]))
        if cat_count >= 1 and num_count >= 1:
            c1, c2 = st.columns(2)
            with c1:
                _auto_bar(chart_df, "Value Breakdown")
            with c2:
                _auto_pie(chart_df, "Distribution")
        with st.expander("View data table", expanded=False):
            st.dataframe(chart_df, use_container_width=True, hide_index=True)

    # Other step dataframes
    for step_num, result in step_dfs.items():
        if result is chart_df:
            continue
        if isinstance(result, pd.DataFrame) and not result.empty:
            with st.expander(f"Step {step_num} data", expanded=False):
                st.dataframe(result, use_container_width=True, hide_index=True)

    for rec in final_state.get("recommendations", []):
        st.info(rec)


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
                    st.dataframe(log["df"], use_container_width=True, hide_index=True)
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
        final_state = {}

        try:
            graph, step_dfs = build_graph(api_key, df)

            initial_state: AgentState = {
                "question": question,
                "plan": "",
                "data_results": [],
                "sufficient": False,
                "metrics": [],
                "recommendations": [],
                "loop_count": 0,
                "error": None,
            }

            # Accumulate the full state across all node updates
            merged_state = dict(initial_state)

            with st.status("🤔 Starting analysis...", expanded=True) as status:
                for event in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        # Merge this node's output into the running state
                        merged_state.update(node_output)

                        label = NODE_LABELS.get(node_name, f"Running {node_name}...")
                        status.update(label=f"⚙️ {label}")

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
                                        st.dataframe(step_df, use_container_width=True, hide_index=True)
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

            final_state = merged_state

        except Exception as e:
            st.error(f"Agent error: {e}")

        if final_state.get("metrics") or final_state.get("recommendations") or step_dfs:
            _render_report(final_state, step_dfs)
            st.session_state["agent_history"].append({
                "question": question,
                "steps_log": steps_log,
                "final_state": final_state,
                "step_dfs": step_dfs,
            })

if st.session_state["agent_history"]:
    if st.button("Clear history"):
        st.session_state["agent_history"] = []
        st.rerun()

render_chat_sidebar()
