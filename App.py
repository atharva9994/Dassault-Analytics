import streamlit as st
from data_cleaner import clean_uploaded_file

st.set_page_config(
    page_title="3DS License Sales Analytics",
    page_icon="🔵",
    layout="wide",
)

st.title("3DS License Sales Analytics")
st.markdown("Upload your sales data to get started. Supports **CSV** and **Excel** files.")

uploaded_file = st.file_uploader(
    "Drag and drop your file here or click to browse",
    type=["csv", "xlsx", "xls"],
    help="CSV or Excel files are supported. Excel files with multiple sheets will be combined.",
)

if uploaded_file is not None:
    with st.spinner("Cleaning and loading data…"):
        df, summary = clean_uploaded_file(uploaded_file)

    if "error" in summary:
        st.error(f"Failed to load file: {summary['error']}")
    else:
        st.session_state["data"] = df
        st.session_state["summary"] = summary

        st.success("File loaded and cleaned successfully!")

        st.subheader("Cleaning Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sheets Found", summary["sheet_count"])
        c2.metric("Rows Before", summary["total_rows_before"])
        c3.metric("Rows After", summary["total_rows_after"])
        c4.metric("Duplicates Removed", summary["duplicates_removed"])
        c5.metric("Blanks Fixed", summary["blanks_fixed"])

        with st.expander("Sheet names & columns detected"):
            st.write("**Sheets:**", ", ".join(summary["sheet_names"]))
            st.write(f"**Columns detected ({summary['columns_detected']}):")
            st.write(", ".join(summary["column_names"]))

        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

elif "data" not in st.session_state:
    st.info("Upload a file to begin. Navigate to the pages in the sidebar once data is loaded.")
else:
    st.info("Data already loaded. Use the sidebar to navigate to the analytics pages.")
    st.dataframe(st.session_state["data"].head(20), use_container_width=True)
