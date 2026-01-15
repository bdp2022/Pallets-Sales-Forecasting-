# app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import chardet

st.title("📦 Pallet Sales Forecasting App (Prophet)")

# --- Upload files ---
hist_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
new_file = st.file_uploader("Upload Future Dates (CSV)", type=["csv"])

# --- Simplified, safe CSV reader ---
def safe_read_csv(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', skip_blank_lines=True)
        if df.empty:
            raise ValueError
    except Exception:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip', skip_blank_lines=True)
    df.columns = df.columns.str.strip()  # remove extra spaces
    return df

# --- Automatic column detection ---
def detect_columns(df, is_future=False):
    cols = [c.upper() for c in df.columns]
    
    # Detect date column
    date_cols = [c for c in cols if 'DATE' in c]
    if not date_cols:
        return None, None
    date_col = df.columns[cols.index(date_cols[0])]
    
    if is_future:
        return date_col, None
    
    # Detect quantity column
    qty_cols = [c for c in cols if any(x in c for x in ['QUANTITY','QTY','VALUE','AMOUNT'])]
    if not qty_cols:
        return date_col, None
    qty_col = df.columns[cols.index(qty_cols[0])]
    
    return date_col, qty_col

if hist_file is not None and new_file is not None:
    # --- Read files safely ---
    hist_data = safe_read_csv(hist_file)
    new_data = safe_read_csv(new_file)

    st.write("### Historical Data Preview")
    st.dataframe(hist_data.head())

    st.write("### New Data Preview")
    st.dataframe(new_data.head())

    # --- Detect columns ---
    hist_date_col, hist_qty_col = detect_columns(hist_data)
    if hist_date_col is None or hist_qty_col is None:
        st.error("Historical data must contain a date column and a quantity column")
        st.stop()
    
    new_date_col, _ = detect_columns(new_data, is_future=True)
    if new_date_col is None:
        st.error("New data must contain a date column")
        st.stop()

    # --- Prepare historical data ---
    df = hist_data.rename(columns={hist_date_col: 'ds', hist_qty_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])
    df['y'] = pd.to_numeric(df['y'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=['y'])

    if df.empty:
        st.error("Historical data has no valid rows after cleaning.")
        st.stop()

    # --- Train Prophet model ---
    model = Prophet()
    model.fit(df)

    # --- Prepare future dataframe ---
    future = new_data.rename(columns={new_date_col: 'ds'})
    future['ds'] = pd.to_datetime(future['ds'], errors='coerce')
    future = future.dropna(subset=['ds'])

    if future.empty:
        st.error("Future dataframe has no rows after parsing. Check your uploaded CSV!")
        st.stop()

    # --- Forecast ---
    forecast = model.predict(future)

    # --- Keep only main forecast as QUANTITY and rename ds to POSTING DATE ---
    forecast_display = forecast[['ds', 'yhat']].rename(columns={
        'ds': 'POSTING DATE',
        'yhat': 'QUANTITY'
    })

    st.write("### Forecasted Results (QUANTITY)")
    st.dataframe(forecast_display)

    # --- Plot forecast (clean) ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(forecast_display['POSTING DATE'], forecast_display['QUANTITY'], marker='o', linestyle='-', color='blue')
    ax.set_title("Forecasted QUANTITY")
    ax.set_xlabel("POSTING DATE")
    ax.set_ylabel("QUANTITY")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # --- Download forecasted results ---
    csv = forecast_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name='forecasted_sales.csv',
        mime='text/csv'
    )
