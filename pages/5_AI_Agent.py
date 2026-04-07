import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px
import json
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("AI Agent")
st.caption("Ask complex multi-step questions. The agent breaks them down, runs real computations, and returns a visual executive report.")

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


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_agent_prompt(df: pd.DataFrame) -> str:
    cols = ", ".join(df.columns.tolist())
    sample = df.head(3).to_string(index=False)
    return f"""You are an AI data analyst for Dassault Systèmes with access to a license sales dataset.

Columns available: {cols}

Sample rows:
{sample}

Reason through the question step by step. For EVERY step respond with ONLY a valid JSON object — no markdown, no text outside the JSON.

Intermediate step format:
{{
  "step_number": 1,
  "step_description": "Short description of what this step computes",
  "pandas_code": "result = df.groupby('Product')['Deal_Value_USD'].sum().reset_index()",
  "is_final": false
}}

Final step format — return structured data, NOT a text summary:
{{
  "step_number": 4,
  "step_description": "Executive Summary",
  "is_final": true,
  "metrics": [
    {{"label": "Total Revenue at Risk", "value": "$2,450,000"}},
    {{"label": "High-Risk Customers", "value": "12"}},
    {{"label": "Medium-Risk Customers", "value": "28"}}
  ],
  "chart_step": 2,
  "warnings": [
    "12 high-risk customers represent $2.4M in immediate churn exposure"
  ],
  "recommendations": [
    "Prioritise CATIA renewal outreach in EMEA — 6 accounts, $1.2M at stake",
    "Offer usage training to low-engagement accounts before renewal dates"
  ]
}}

Rules:
- Respond ONLY with a valid JSON object, nothing else
- pandas_code must store output in a variable named `result`
- result must be a pandas DataFrame (max 10 rows, max 5 columns) or a scalar
- Use exact column names from the dataset
- Revenue = Deal_Value_USD column
- metrics must use real numbers from step results — never generic placeholders
- warnings: 1–3 urgent, specific, data-backed alerts
- recommendations: 2–4 short actionable bullets (1–2 lines each)
- chart_step: the step number whose result dataframe is most suitable for a bar chart
- 2 to 5 steps is sufficient"""


# ── JSON parser ───────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        return json.loads(obj_match.group())
    raise ValueError(f"No JSON object found in: {raw[:300]}")


# ── Chart helper ──────────────────────────────────────────────────────────────
def _auto_bar(result_df: pd.DataFrame, title: str) -> None:
    """Plot the first categorical + first numeric column as a horizontal bar chart."""
    if result_df is None or result_df.empty:
        return
    cat_cols = [c for c in result_df.columns if result_df[c].dtype == object]
    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    if not cat_cols or not num_cols:
        return
    fig = px.bar(
        result_df,
        x=num_cols[0],
        y=cat_cols[0],
        orientation="h",
        title=title,
        color=num_cols[0],
        color_continuous_scale="Blues",
        text=num_cols[0],
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin={"t": 40, "b": 20},
        height=max(300, len(result_df) * 40 + 80),
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def _auto_pie(result_df: pd.DataFrame, title: str) -> None:
    """Plot a pie chart from a two-column df (label + value)."""
    if result_df is None or result_df.empty:
        return
    cat_cols = [c for c in result_df.columns if result_df[c].dtype == object]
    num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
    if not cat_cols or not num_cols:
        return
    fig = px.pie(result_df, names=cat_cols[0], values=num_cols[0], title=title,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(margin={"t": 40, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


# ── Final visual renderer ─────────────────────────────────────────────────────
def _render_final(final_event: dict, step_results: dict) -> None:
    """Render the final structured JSON as a professional visual report."""
    st.markdown("---")
    st.markdown(f"### {final_event['description']}")

    # Metrics row
    metrics = final_event.get("metrics", [])
    if metrics:
        metric_cols = st.columns(len(metrics))
        for col, m in zip(metric_cols, metrics):
            col.metric(m.get("label", ""), m.get("value", ""))
        st.markdown("")

    # Warnings
    for w in final_event.get("warnings", []):
        st.warning(w)

    # Charts — pick the chart_step dataframe, plus the first other df for a pie
    chart_step = final_event.get("chart_step")
    chart_df = step_results.get(chart_step) if chart_step else None

    # Fallback: use the largest DataFrame available
    if chart_df is None or not isinstance(chart_df, pd.DataFrame) or chart_df.empty:
        for df_candidate in step_results.values():
            if isinstance(df_candidate, pd.DataFrame) and not df_candidate.empty:
                chart_df = df_candidate
                break

    if chart_df is not None and isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
        num_cols_count = sum(1 for c in chart_df.columns if pd.api.types.is_numeric_dtype(chart_df[c]))
        cat_cols_count = sum(1 for c in chart_df.columns if chart_df[c].dtype == object)

        if cat_cols_count >= 1 and num_cols_count >= 1:
            c1, c2 = st.columns(2)
            with c1:
                _auto_bar(chart_df, "Revenue / Value Breakdown")
            with c2:
                _auto_pie(chart_df, "Distribution")

        # Show as formatted table below charts
        with st.expander("View data table", expanded=False):
            st.dataframe(chart_df, use_container_width=True, hide_index=True)

    # All other step dataframes as small expanders
    for step_num, result in step_results.items():
        if result is chart_df:
            continue
        if isinstance(result, pd.DataFrame) and not result.empty:
            with st.expander(f"Step {step_num} data", expanded=False):
                st.dataframe(result, use_container_width=True, hide_index=True)

    # Recommendations
    recs = final_event.get("recommendations", [])
    if recs:
        st.markdown("**Recommendations**")
        for rec in recs:
            st.info(rec)


# ── Agent runner ──────────────────────────────────────────────────────────────
STEP_ICONS = {1: "📊", 2: "⚠️", 3: "🔍", 4: "💡", 5: "📈", 6: "🎯"}

def run_agent(question: str, df: pd.DataFrame, api_key: str):
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_agent_prompt(df)
    messages = [{"role": "user", "content": question}]

    for _ in range(7):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": raw})

        try:
            step = _parse_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            yield {"type": "error", "message": f"Parse error: {e}\n\nRaw: {raw[:300]}"}
            return

        step_num = step.get("step_number", len(messages) // 2)

        if step.get("is_final"):
            yield {
                "type": "final",
                "number": step_num,
                "description": step.get("step_description", "Executive Summary"),
                "metrics": step.get("metrics", []),
                "chart_step": step.get("chart_step"),
                "warnings": step.get("warnings", []),
                "recommendations": step.get("recommendations", []),
            }
            return

        result = None
        error_msg = None
        code = step.get("pandas_code", "")
        if code:
            try:
                local_vars = {"df": df.copy(), "pd": pd}
                exec(code, {}, local_vars)
                result = local_vars.get("result", None)
            except Exception as e:
                error_msg = str(e)

        yield {
            "type": "step",
            "number": step_num,
            "description": step.get("step_description", f"Step {step_num}"),
            "result": result,
            "error": error_msg,
        }

        if result is not None and isinstance(result, pd.DataFrame):
            result_str = result.head(10).to_string(index=False)
        elif result is not None:
            result_str = str(result)
        else:
            result_str = f"Execution error: {error_msg}" if error_msg else "No result produced."

        messages.append({
            "role": "user",
            "content": f"Step {step_num} result:\n{result_str}\n\nContinue to the next step.",
        })


# ── Replay helper ─────────────────────────────────────────────────────────────
def _replay_entry(entry: dict) -> None:
    step_results = {}
    final_event = None
    for s in entry["steps"]:
        if s["type"] == "step":
            icon = STEP_ICONS.get(s["number"], "🔹")
            suffix = f" — {len(s['result'])} rows" if isinstance(s.get("result"), pd.DataFrame) else ""
            with st.expander(f"{icon} Step {s['number']}: {s['description']}{suffix}", expanded=False):
                if s.get("error"):
                    st.warning(s["error"])
                elif isinstance(s.get("result"), pd.DataFrame):
                    st.dataframe(s["result"], use_container_width=True, hide_index=True)
                elif s.get("result") is not None:
                    st.metric("Result", s["result"])
            if isinstance(s.get("result"), pd.DataFrame):
                step_results[s["number"]] = s["result"]
        elif s["type"] == "final":
            final_event = s
        elif s["type"] == "error":
            st.error(s["message"])
    if final_event:
        _render_final(final_event, step_results)


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
        _replay_entry(entry)

# ── Run agent ─────────────────────────────────────────────────────────────────
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        completed_steps = []
        step_results = {}

        with st.status("🤔 Understanding your question...", expanded=True) as status:
            try:
                for event in run_agent(question, df, api_key):
                    completed_steps.append(event)
                    icon = STEP_ICONS.get(event.get("number", 1), "🔹")

                    if event["type"] == "step":
                        result = event.get("result")
                        suffix = f" — {len(result)} rows" if isinstance(result, pd.DataFrame) else ""
                        status.update(label=f"{icon} Step {event['number']}: {event['description']}{suffix}")

                        with st.expander(
                            f"{icon} Step {event['number']}: {event['description']}{suffix}",
                            expanded=False,
                        ):
                            if event.get("error"):
                                st.warning(event["error"])
                            elif isinstance(result, pd.DataFrame):
                                st.dataframe(result, use_container_width=True, hide_index=True)
                            elif result is not None:
                                st.metric("Result", result)

                        if isinstance(result, pd.DataFrame):
                            step_results[event["number"]] = result

                    elif event["type"] == "final":
                        status.update(label="✅ Analysis complete", state="complete")

                    elif event["type"] == "error":
                        status.update(label="Error during analysis", state="error")
                        st.error(event["message"])

            except anthropic.AuthenticationError:
                status.update(label="Authentication failed", state="error")
                st.error("Invalid API key.")
            except Exception as e:
                status.update(label="Unexpected error", state="error")
                st.error(f"Error: {e}")

        # Render final visual report outside the status box
        for event in completed_steps:
            if event["type"] == "final":
                _render_final(event, step_results)

        if completed_steps:
            st.session_state["agent_history"].append({
                "question": question,
                "steps": completed_steps,
            })

if st.session_state["agent_history"]:
    if st.button("Clear history"):
        st.session_state["agent_history"] = []
        st.rerun()

render_chat_sidebar()
