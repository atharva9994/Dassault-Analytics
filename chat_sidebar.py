import streamlit as st
import anthropic
import pandas as pd

_CODE_SEP = "<<<CODE>>>"

_SYSTEM_PROMPT_TEMPLATE = """You are a senior business analyst assistant for Dassault Systemes license sales.
The user has a pandas DataFrame named `df` with these columns:
{col_info}

Sample rows:
{sample_rows}

== RULES ==

1. ALWAYS start with a plain-English answer inside <explanation> tags:
   - 1-2 business-focused sentences (no jargon, no mention of Python/code/pandas)
   - Exactly 3 bullet points starting with a dash -, each a specific finding or number
   - Format all currency as $1,234,567 inside the explanation

2. For ANY question that involves data — rankings, totals, counts, customers, segments,
   revenue, churn, products, regions, performance — you MUST append {sep} followed by
   Python code. Only skip code for purely conceptual/advisory questions with no data
   component (e.g. "give ideas", "explain a concept").
   - If you do write code: produce exactly ONE result stored in `result`:
       * A clean pandas DataFrame with at most 10 rows and at most 5 columns, OR
       * A single scalar (int or float)
   - NEVER return dicts, tuples, lists, or nested structures.
   - NEVER assign multiple DataFrames. No complex multi-step analysis.
   - Column names must be human-readable (e.g. "Revenue" not "Deal_Value_USD").
   - Round all floats to 2 decimal places. No scientific notation.

3. Response format (follow exactly):

<explanation>
One or two plain sentences summarising the answer.
- Finding 1 with a specific number or fact
- Finding 2 with a specific number or fact
- Finding 3 with a specific number or fact
</explanation>
{sep}
result = df.groupby("Product")["Deal_Value_USD"].sum().reset_index()

If NO code is needed, stop after </explanation> with no separator.
"""


def _build_system_prompt(df: pd.DataFrame) -> str:
    col_info = ", ".join(df.columns.tolist())
    sample_rows = df.head(3).to_string(index=False)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        col_info=col_info,
        sample_rows=sample_rows,
        sep=_CODE_SEP,
    )


def _parse_response(raw: str) -> tuple[str, str]:
    """Return (explanation, code). Code is empty string for conceptual answers."""
    code = ""
    explanation_block = raw

    if _CODE_SEP in raw:
        explanation_block, code = raw.split(_CODE_SEP, 1)

    for tag in ("<explanation>", "</explanation>"):
        explanation_block = explanation_block.replace(tag, "")

    return explanation_block.strip(), code.strip()


def _format_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for clean display."""
    out = df.copy()
    for col in out.columns:
        col_lower = col.lower()
        if pd.api.types.is_float_dtype(out[col]):
            if any(k in col_lower for k in ["revenue", "value", "amount", "sales", "deal"]):
                out[col] = out[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            else:
                out[col] = out[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].apply(lambda x: f"{x:,}")
    return out


def _render_result(result, sidebar: bool = False) -> None:
    """Render a result cleanly. Silently skips dicts, raw dumps, unrenderable types."""
    ctx = st.sidebar if sidebar else st

    if isinstance(result, pd.DataFrame):
        if not result.empty:
            ctx.dataframe(_format_df_for_display(result), use_container_width=True, hide_index=True)
        return

    if isinstance(result, pd.Series):
        frame = result.reset_index()
        frame.columns = [str(c) for c in frame.columns]
        ctx.dataframe(_format_df_for_display(frame), use_container_width=True, hide_index=True)
        return

    if isinstance(result, (int, float)) and not isinstance(result, bool):
        display = f"${result:,.0f}" if result > 10_000 else f"{result:,.2f}"
        ctx.metric(label="Result", value=display)
        return

    if isinstance(result, str) and len(result) < 300:
        ctx.info(result)
        return

    # Dicts, tuples, complex objects, long strings → silently drop (never show raw dumps)
    return


def render_chat_sidebar():
    """Render the Ask Your Data chat panel in the sidebar."""
    if "data" not in st.session_state:
        return

    df = st.session_state["data"]

    st.sidebar.divider()
    st.sidebar.header("Ask Your Data")

    # ── API key: secrets → session_state → manual fallback ───────────────────
    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY", "")
        or st.session_state.get("anthropic_api_key", "")
    )

    if not api_key:
        manual = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            key="sidebar_api_key_input",
        )
        if manual:
            st.session_state["anthropic_api_key"] = manual
            api_key = manual
        else:
            st.sidebar.caption("Enter your API key above to enable chat.")
            return

    # ── Chat history ──────────────────────────────────────────────────────────
    if "sidebar_chat" not in st.session_state:
        st.session_state["sidebar_chat"] = []

    # ── Question input ────────────────────────────────────────────────────────
    question = st.sidebar.text_input(
        "Ask a question",
        placeholder="e.g. Who are the top 5 customers by revenue?",
        key="sidebar_chat_input",
    )
    ask_clicked = st.sidebar.button("Ask", use_container_width=True)

    if ask_clicked and question.strip():
        with st.sidebar:
            with st.spinner("Thinking..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    system_prompt = _build_system_prompt(df)

                    messages = []
                    for entry in st.session_state["sidebar_chat"]:
                        messages.append({"role": "user",      "content": entry["q"]})
                        messages.append({"role": "assistant", "content": entry["raw"]})
                    messages.append({"role": "user", "content": question.strip()})

                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages,
                    )
                    raw = response.content[0].text.strip()
                    explanation, code = _parse_response(raw)

                    result = None
                    if code:
                        local_vars = {"df": df.copy(), "pd": pd}
                        try:
                            exec(code, {}, local_vars)
                            result = local_vars.get("result", None)
                        except Exception:
                            result = None  # silently drop — explanation already shown

                    st.session_state["sidebar_chat"].append({
                        "q":           question.strip(),
                        "raw":         raw,
                        "explanation": explanation,
                        "result":      result,
                    })

                except anthropic.AuthenticationError:
                    st.sidebar.error("Invalid API key.")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    # ── Display history (newest first) ────────────────────────────────────────
    for entry in reversed(st.session_state["sidebar_chat"]):
        st.sidebar.markdown(f"**{entry['q']}**")
        if entry.get("explanation"):
            st.sidebar.markdown(entry["explanation"])
        if entry.get("result") is not None:
            _render_result(entry["result"], sidebar=True)
        st.sidebar.divider()

    if st.session_state["sidebar_chat"]:
        if st.sidebar.button("Clear history", use_container_width=True):
            st.session_state["sidebar_chat"] = []
            st.rerun()
