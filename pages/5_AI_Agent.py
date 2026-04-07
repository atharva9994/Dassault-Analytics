import streamlit as st
import anthropic
import pandas as pd
import json
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import render_chat_sidebar

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("AI Agent")
st.caption("Ask complex multi-step questions. The agent breaks them down, runs real computations, and returns data-driven recommendations.")

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

When given a business question, reason through it step by step. For EVERY step respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.

Intermediate step format:
{{
  "step_number": 1,
  "step_description": "Short description of what this step computes",
  "pandas_code": "result = df.groupby('Product')['Deal_Value_USD'].sum().reset_index()",
  "is_final": false
}}

Final step format (when you have enough data to give conclusions):
{{
  "step_number": 4,
  "step_description": "Final Analysis and Recommendations",
  "summary": "Detailed findings with specific numbers from the data and concrete recommendations.",
  "is_final": true
}}

Critical rules:
- Respond ONLY with a JSON object, nothing else
- pandas_code must assign output to a variable named `result`
- result must be a pandas DataFrame (max 10 rows, max 5 columns) or a single scalar
- Use exact column names from the dataset above
- Base every recommendation on actual numbers returned from the data — never give generic advice
- Revenue = Deal_Value_USD column
- 2 to 5 steps is sufficient for most questions"""


# ── JSON parser (robust to ```json``` wrapping) ───────────────────────────────
def _parse_json(raw: str) -> dict:
    text = raw.strip()
    # Strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    # Extract first {...} block
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        return json.loads(obj_match.group())
    raise ValueError(f"No JSON object found in: {raw[:300]}")


# ── Agent runner ──────────────────────────────────────────────────────────────
STEP_ICONS = {1: "📊", 2: "⚠️", 3: "🔍", 4: "💡", 5: "📈", 6: "🎯"}

def run_agent(question: str, df: pd.DataFrame, api_key: str):
    """
    Generator that yields dicts:
      {"type": "step",  "number": N, "description": str, "result": df|scalar|None, "error": str|None}
      {"type": "final", "number": N, "description": str, "summary": str}
      {"type": "error", "message": str}
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_agent_prompt(df)
    messages = [{"role": "user", "content": question}]

    for _ in range(7):  # max iterations guard
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
            yield {"type": "error", "message": f"Parse error: {e}\n\nRaw response: {raw[:300]}"}
            return

        step_num = step.get("step_number", len(messages) // 2)

        if step.get("is_final"):
            yield {
                "type": "final",
                "number": step_num,
                "description": step.get("step_description", "Final Analysis"),
                "summary": step.get("summary", ""),
            }
            return

        # Execute pandas code
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

        # Summarise result to feed back to Claude
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


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "Which CATIA customers are at churn risk and what should we do?",
    "Compare subscription vs perpetual revenue trends and recommend a strategy",
    "Find underperforming regions and suggest growth opportunities",
    "Identify our top 10 customers and analyze their license patterns",
]

st.markdown("**Example questions:**")
cols = st.columns(len(EXAMPLES))
selected_example = None
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        selected_example = example

# ── Question input ────────────────────────────────────────────────────────────
question = st.chat_input("Ask a complex business question about your data...")
if selected_example:
    question = selected_example

# ── Session history ───────────────────────────────────────────────────────────
if "agent_history" not in st.session_state:
    st.session_state["agent_history"] = []

# Replay previous results
for entry in st.session_state["agent_history"]:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        for s in entry["steps"]:
            icon = STEP_ICONS.get(s["number"], "🔹")
            if s["type"] == "step":
                suffix = ""
                if s.get("result") is not None and isinstance(s["result"], pd.DataFrame):
                    suffix = f" — {len(s['result'])} rows"
                elif s.get("result") is not None:
                    suffix = f" — result: {s['result']}"
                with st.expander(f"{icon} Step {s['number']}: {s['description']}{suffix}", expanded=False):
                    if s.get("error"):
                        st.warning(f"Step encountered an issue: {s['error']}")
                    elif s.get("result") is not None:
                        if isinstance(s["result"], pd.DataFrame):
                            st.dataframe(s["result"], use_container_width=True, hide_index=True)
                        else:
                            st.metric("Result", s["result"])
            elif s["type"] == "final":
                st.success(f"**{icon} {s['description']}**\n\n{s['summary']}")
            elif s["type"] == "error":
                st.error(s["message"])

# ── Run agent on new question ─────────────────────────────────────────────────
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        completed_steps = []

        with st.status("🤔 Understanding your question...", expanded=True) as status:
            try:
                for event in run_agent(question, df, api_key):
                    icon = STEP_ICONS.get(event.get("number", 1), "🔹")
                    completed_steps.append(event)

                    if event["type"] == "step":
                        suffix = ""
                        result = event.get("result")
                        if isinstance(result, pd.DataFrame):
                            suffix = f" — {len(result)} rows"
                        elif result is not None:
                            suffix = f" — result: {result}"

                        label = f"{icon} Step {event['number']}: {event['description']}{suffix}"
                        status.update(label=label)

                        if event.get("error"):
                            st.warning(f"Step {event['number']} issue: {event['error']}")
                        elif result is not None:
                            st.markdown(f"**Step {event['number']}:** {event['description']}")
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result, use_container_width=True, hide_index=True)
                            else:
                                st.metric(f"Step {event['number']} result", result)

                    elif event["type"] == "final":
                        status.update(label=f"💡 {event['description']}", state="complete")

                    elif event["type"] == "error":
                        status.update(label="Analysis encountered an error", state="error")
                        st.error(event["message"])

            except anthropic.AuthenticationError:
                status.update(label="Authentication failed", state="error")
                st.error("Invalid API key.")
            except Exception as e:
                status.update(label="Unexpected error", state="error")
                st.error(f"Error: {e}")

        # Show final summary outside status box
        for event in completed_steps:
            if event["type"] == "final":
                st.success(f"**💡 {event['description']}**\n\n{event['summary']}")

        # Save to history
        if completed_steps:
            st.session_state["agent_history"].append({
                "question": question,
                "steps": completed_steps,
            })

if st.session_state["agent_history"]:
    if st.button("Clear history", use_container_width=False):
        st.session_state["agent_history"] = []
        st.rerun()

# ── Sidebar chat ──────────────────────────────────────────────────────────────
render_chat_sidebar()
