import pandas as pd
import io

COLUMN_NAME_MAP = {
    "revenue": "Deal_Value_USD",
    "deal_value": "Deal_Value_USD",
    "sales amount": "Deal_Value_USD",
    "sales_amount": "Deal_Value_USD",
    "deal value": "Deal_Value_USD",
    "amount": "Deal_Value_USD",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized in COLUMN_NAME_MAP:
            rename_map[col] = COLUMN_NAME_MAP[normalized]
    return df.rename(columns=rename_map)


def clean_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, dict]:
    try:
        filename = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()

        sheet_names = []
        sheet_count = 0

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
            sheet_count = 1
            sheet_names = ["Sheet1"]
        elif filename.endswith((".xlsx", ".xls")):
            excel = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = excel.sheet_names
            sheet_count = len(sheet_names)
            frames = [excel.parse(sheet) for sheet in sheet_names]
            df = pd.concat(frames, ignore_index=True)
        else:
            return None, {"error": f"Unsupported file type: {uploaded_file.name}"}

        total_rows_before = len(df)

        # Standardize column names
        df = _standardize_columns(df)

        # Remove duplicate rows based on Deal_ID if present
        duplicates_removed = 0
        if "Deal_ID" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["Deal_ID"])
            duplicates_removed = before - len(df)

        # Convert Deal_Value_USD to numeric (strip $ and commas)
        if "Deal_Value_USD" in df.columns:
            df["Deal_Value_USD"] = (
                df["Deal_Value_USD"]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
            )
            df["Deal_Value_USD"] = pd.to_numeric(df["Deal_Value_USD"], errors="coerce")

        # Convert Booking_Date to datetime
        if "Booking_Date" in df.columns:
            df["Booking_Date"] = pd.to_datetime(df["Booking_Date"], errors="coerce")

        # Fill missing values
        blanks_fixed = int(df.isna().sum().sum())
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(0)

        total_rows_after = len(df)

        summary = {
            "sheet_count": sheet_count,
            "sheet_names": sheet_names,
            "total_rows_before": total_rows_before,
            "total_rows_after": total_rows_after,
            "duplicates_removed": duplicates_removed,
            "blanks_fixed": blanks_fixed,
            "columns_detected": len(df.columns),
            "column_names": list(df.columns),
        }

        return df, summary

    except Exception as e:
        return None, {"error": str(e)}
