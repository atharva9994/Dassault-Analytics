import streamlit as st
import anthropic
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from chat_sidebar import _parse_response, _render_result

_CODE_SEP = "<<<CODE>>>"

def _build_ask_data_prompt(df: pd.DataFrame) -> str:
    col_info = ", ".join(df.columns.tolist())
    sample_rows = df.head(3).to_string(index=False)
    return f"""You are a senior business analyst assistant for Dassault Systemes license sales.
The user has a pandas DataFrame named `df` with these columns:
{col_info}

Sample rows:
{sample_rows}

== RULES ==

1. ALWAYS start with a plain-English answer inside <explanation> tags:
   - 1-2 business-focused sentences
   - Exactly 3 bullet points starting with a dash -, each a specific finding or number from the computed result
   - Format all currency as $1,234,567

2. For ANY question that involves data — rankings, totals, counts, customers, segments, revenue,
   churn, products, regions, performance — you MUST append {_CODE_SEP} followed by Python code.
   Only skip code for purely conceptual/advisory questions with no data component.

   Code rules:
   - Produce exactly ONE variable named `result`
   - result must be a clean pandas DataFrame (≤10 rows, ≤5 columns) or a single scalar
   - Column names must be human-readable (e.g. "Revenue" not "Deal_Value_USD")
   - Round all floats to 2 decimal places
   - NEVER return dicts, lists, tuples, or nested structures
   - Use the real `df` — do NOT hardcode values

3. Response format (follow exactly):

<explanation>
One or two plain sentences summarising the answer.
- Finding 1 with a specific number or fact
- Finding 2 with a specific number or fact
- Finding 3 with a specific number or fact
</explanation>
{_CODE_SEP}
result = df.groupby("Customer")["Deal_Value_USD"].sum().nlargest(5).reset_index().rename(columns={{"Deal_Value_USD": "Revenue"}})

If NO code is needed, stop after </explanation> with no separator.
"""

st.set_page_config(page_title="Ask Data", page_icon="💬", layout="wide")
st.title("Ask Your Data")
st.caption("Ask business questions in plain English — get instant insights from your sales data.")

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
        st.info("Enter your API key above to enable chat.")
        st.stop()

# ── Sample question buttons ───────────────────────────────────────────────────
_SAMPLE_QUESTIONS = [
    "What are the top 5 customers by revenue?",
    "Compare CATIA vs SOLIDWORKS sales by region",
    "Show marketing campaign ROI by channel",
    "Which marketing channel generates the most leads?",
]
scols = st.columns(len(_SAMPLE_QUESTIONS))
for col, sq in zip(scols, _SAMPLE_QUESTIONS):
    if col.button(sq, use_container_width=True):
        st.session_state["_sample_q"] = sq
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Render existing messages
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg.get("explanation"):
                st.markdown(msg["explanation"])
            if msg.get("result") is not None:
                _render_result(msg["result"], sidebar=False)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your data...")
if not user_input and st.session_state.get("_sample_q"):
    user_input = st.session_state.pop("_sample_q")

if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                system_prompt = _build_ask_data_prompt(df)

                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chat_history"]
                ]
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=messages,
                )
                raw = response.content[0].text.strip()
                explanation, code = _parse_response(raw)

                result = None
                q_lower = user_input.lower()
                if any(kw in q_lower for kw in ("segment", "segmentation")) and "Customer_Segment" in df.columns:
                    result = (
                        df.groupby("Customer_Segment")
                        .agg(Total_Revenue=("Deal_Value_USD", "sum"), Customer_Count=("Deal_Value_USD", "count"))
                        .reset_index()
                        .sort_values("Total_Revenue", ascending=False)
                        .rename(columns={"Customer_Segment": "Segment"})
                    )
                    result["Total_Revenue"] = result["Total_Revenue"].round(2)
                elif code:
                    local_vars = {"df": df.copy(), "pd": pd}
                    try:
                        exec(code, {}, local_vars)
                        result = local_vars.get("result", None)
                    except Exception:
                        result = None  # explanation is enough; never show error dumps

                if explanation:
                    st.markdown(explanation)
                if result is not None:
                    _render_result(result, sidebar=False)

                st.session_state["chat_history"].append({
                    "role":        "assistant",
                    "content":     raw,
                    "explanation": explanation,
                    "result":      result,
                })

            except anthropic.AuthenticationError:
                st.error("Invalid API key.")
            except Exception as e:
                st.error(f"Error: {e}")
