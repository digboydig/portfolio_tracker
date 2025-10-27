import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import io

# --- Page config ---
st.set_page_config(
    page_title="Real-Time Portfolio Tracker (NSE/BSE Ready)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Defaults ---
ASIA_KOLKATA = ZoneInfo("Asia/Kolkata")
DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "^NSEI"],
    "Shares": [50.0, 20.0, 10.0, 0.0],  # indices/watchlist with 0 shares
    "Avg Cost": [2450.00, 3200.00, 1500.00, 0.00]
})

# --- Session state for persistence ---
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = DEFAULT_PORTFOLIO.copy()

if "last_updated" not in st.session_state:
    st.session_state.last_updated = None

# --- Helpers ---
def format_currency(value):
    return f"₹{value:,.2f}"

def now_kolkata():
    return datetime.now(tz=ASIA_KOLKATA)

def normalize_ticker_list(tickers):
    seen = set()
    out = []
    for t in tickers:
        tt = str(t).strip()
        if tt and tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out

# --- Robust data fetcher ---
@st.cache_data(ttl=60)
def get_stock_data(tickers_tuple, start_date_str):
    tickers = list(tickers_tuple)
    fetch_warnings = []

    current_prices = pd.Series(index=tickers, dtype='float64')
    prev_closes = pd.Series(index=tickers, dtype='float64')
    history = None

    def as_series_from_val(val, tickers):
        if isinstance(val, pd.Series):
            try:
                return val.reindex(tickers)
            except Exception:
                return pd.Series(dtype='float64', index=tickers)
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return pd.Series(dtype='float64', index=tickers)
            last_row = val.iloc[-1]
            try:
                return last_row.reindex(tickers)
            except Exception:
                return pd.Series(dtype='float64', index=tickers)
        if np.isscalar(val):
            if len(tickers) == 1:
                return pd.Series({tickers[0]: float(val)})
            else:
                return pd.Series(dtype='float64', index=tickers)
        return pd.Series(dtype='float64', index=tickers)

    # Intraday latest price
    try:
        intraday = yf.download(tickers, period="1d", interval="1m", progress=False, auto_adjust=True)
        if isinstance(intraday, pd.DataFrame) and not intraday.empty:
            close_data = None
            if isinstance(intraday.columns, pd.MultiIndex):
                if 'Close' in intraday.columns.get_level_values(0):
                    close_data = intraday['Close'].iloc[-1]
            else:
                if 'Close' in intraday.columns:
                    close_data = intraday['Close'].iloc[-1]
            if close_data is not None:
                s = as_series_from_val(close_data, tickers)
                current_prices.update(s)
        else:
            fetch_warnings.append("Intraday data not available (market closed or API delay).")
    except Exception as e:
        fetch_warnings.append(f"Intraday fetch error: {e}")

    # Previous close
    try:
        prev = yf.download(tickers, period="3d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(prev, pd.DataFrame) and not prev.empty:
            pc = None
            if isinstance(prev.columns, pd.MultiIndex):
                if 'Close' in prev.columns.get_level_values(0):
                    pc = prev['Close'].iloc[-1]
            else:
                if 'Close' in prev.columns:
                    pc = prev['Close'].iloc[-1]
            if pc is not None:
                s = as_series_from_val(pc, tickers)
                prev_closes.update(s)
        else:
            fetch_warnings.append("Previous close data not available (holiday or API delay).")
    except Exception as e:
        fetch_warnings.append(f"Previous-close fetch error: {e}")

    # Historical data
    try:
        hist_raw = yf.download(tickers, start=start_date_str, progress=False, auto_adjust=True)
        if isinstance(hist_raw, pd.DataFrame) and not hist_raw.empty:
            if isinstance(hist_raw.columns, pd.MultiIndex):
                lvl0 = hist_raw.columns.get_level_values(0)
                if 'Adj Close' in lvl0:
                    history = hist_raw['Adj Close'].copy()
                elif 'Close' in lvl0:
                    history = hist_raw['Close'].copy()
                    fetch_warnings.append("Using 'Close' for historical data because 'Adj Close' was unavailable.")
            else:
                if 'Adj Close' in hist_raw.columns:
                    history = hist_raw['Adj Close'].to_frame(name=tickers[0]) if len(tickers) == 1 else hist_raw[['Adj Close']]
                elif 'Close' in hist_raw.columns:
                    history = hist_raw[['Close']].copy() if len(tickers) > 1 else hist_raw['Close'].to_frame(name=tickers[0])
                    fetch_warnings.append("Using 'Close' for historical data because 'Adj Close' was unavailable.")
            if isinstance(history, pd.DataFrame):
                if isinstance(history.columns, pd.MultiIndex):
                    history.columns = history.columns.get_level_values(-1)
                history = history.dropna(axis=1, how='all')
        else:
            fetch_warnings.append("Historical data fetch returned empty result.")
    except Exception as e:
        fetch_warnings.append(f"Historical fetch error: {e}")

    real_time_df = pd.DataFrame({
        "Current Price": current_prices,
        "Previous Close": prev_closes
    })
    real_time_df = real_time_df.reindex(tickers)

    return real_time_df, history, fetch_warnings

# --- Sidebar: Settings & Import/Export ---
with st.sidebar:
    st.header("Settings")
    st.write("Data cache TTL: 60s (adjustable in code)")

    st.markdown("### Portfolio import/export")
    uploaded = st.file_uploader("Upload CSV to replace portfolio (columns: Ticker,Shares,Avg Cost)", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            df_upload = df_upload.rename(columns={c: c.strip() for c in df_upload.columns})
            required = {'Ticker', 'Shares', 'Avg Cost'}
            if not required.issubset(set(df_upload.columns)):
                st.error(f"CSV must contain columns: {required}")
            else:
                st.session_state.portfolio_df = df_upload[['Ticker','Shares','Avg Cost']].copy()
                st.success("Portfolio imported into session.")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

    st.download_button(
        label="Download current portfolio CSV",
        data=st.session_state.portfolio_df.to_csv(index=False),
        file_name="portfolio_export.csv",
        mime="text/csv"
    )

    st.markdown("---")
    # Refresh button: clear cache and attempt to rerun in a compatible way
    if st.button("Refresh Data (clear cache)"):
        st.cache_data.clear()
        # Try the Streamlit rerun APIs in a compatible way
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
        else:
            # Fallback: inform the user to manually refresh if rerun API not available
            st.warning("Auto-rerun not available in this Streamlit version. Please refresh the browser page to reload data.")
            # stop further execution for clarity
            st.stop()

    st.markdown("---")
    timeframe_options = {
        "1 Year": 365,
        "6 Months": 180,
        "3 Months": 90,
        "1 Month": 30
    }
    selected_timeframe = st.selectbox("Historical timeframe", list(timeframe_options.keys()))
    days_to_lookback = timeframe_options[selected_timeframe]
    start_date = (now_kolkata() - timedelta(days=days_to_lookback)).strftime('%Y-%m-%d')
    st.write(f"Showing from: {start_date}")

# --- Main App UI ---
st.title("Real-Time Portfolio Dashboard (NSE/BSE Ready)")
st.caption("Use Yahoo Finance tickers (e.g., RELIANCE.NS, ^NSEI). Data may be delayed during market hours.")

st.subheader("1 — Your Current Holdings (edit and press Enter to apply)")

edited = st.data_editor(
    st.session_state.portfolio_df,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="e.g., TCS.NS or ^NSEI", required=True),
        "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.2f"),
        "Avg Cost": st.column_config.NumberColumn("Avg Cost (₹)", min_value=0.0, format="₹%.2f")
    },
    num_rows="dynamic",
    hide_index=True
)

st.session_state.portfolio_df = edited.copy()

tickers = normalize_ticker_list(st.session_state.portfolio_df['Ticker'].tolist())
if not tickers:
    st.info("Add at least one ticker to continue.")
    st.stop()

real_time_df, history, warnings = get_stock_data(tuple(tickers), start_date)

missing_current = real_time_df[real_time_df['Current Price'].isna()].index.tolist()
if missing_current:
    st.warning(f"No intraday price for: {', '.join(missing_current)}. Using previous close where available.")

for w in warnings:
    st.info(w)

if real_time_df is None:
    st.error("Could not fetch real-time data. Check tickers and network.")
    st.stop()

# --- Portfolio calculations ---
def calculate_portfolio_metrics(portfolio_df, real_time_df):
    current_data = real_time_df.reset_index().rename(columns={'index': 'Ticker'})
    merged = pd.merge(portfolio_df, current_data, on='Ticker', how='left')

    merged['Current Price'] = merged['Current Price'].fillna(merged['Previous Close'])
    merged['Previous Close'] = merged['Previous Close'].fillna(merged['Current Price'])

    merged['Shares'] = pd.to_numeric(merged['Shares'], errors='coerce').fillna(0.0)
    merged['Avg Cost'] = pd.to_numeric(merged['Avg Cost'], errors='coerce').fillna(0.0)

    merged['Cost Basis'] = merged['Shares'] * merged['Avg Cost']
    merged['Current Value'] = merged['Shares'] * merged['Current Price']
    merged['Total Gain/Loss'] = merged['Current Value'] - merged['Cost Basis']
    merged['Daily Change ₹'] = merged['Shares'] * (merged['Current Price'] - merged['Previous Close'])

    merged['Total Return %'] = np.where(merged['Cost Basis'] != 0,
                                       (merged['Total Gain/Loss'] / merged['Cost Basis']) * 100, 0.0)
    merged['Daily Change %'] = np.where(merged['Previous Close'] != 0,
                                       ((merged['Current Price'] - merged['Previous Close']) / merged['Previous Close']) * 100,
                                       0.0)

    holdings = merged[merged['Shares'] > 0].copy()
    total_cost = holdings['Cost Basis'].sum()
    total_value = holdings['Current Value'].sum()
    total_gain = holdings['Total Gain/Loss'].sum()
    total_daily_change = holdings['Daily Change ₹'].sum()
    overall_return_pct = (total_gain / total_cost) * 100 if total_cost else 0.0
    prev_portfolio_value = total_value - total_daily_change
    overall_daily_pct = (total_daily_change / prev_portfolio_value) * 100 if prev_portfolio_value else 0.0

    return merged, total_value, total_gain, overall_return_pct, total_daily_change, overall_daily_pct

portfolio_details, total_value, total_gain_loss, overall_return_percent, total_daily_change, overall_daily_percent = calculate_portfolio_metrics(st.session_state.portfolio_df, real_time_df)

st.session_state.last_updated = now_kolkata()

# --- Display top-level metrics ---
st.subheader("2 — Portfolio Summary (excluding 0-share watchlist items)")
col1, col2, col3, col4 = st.columns([2,2,2,3])  # give more width to the pie column

with col1:
    st.metric("Total Portfolio Value (INR)", format_currency(total_value),
              delta=f"{format_currency(total_daily_change)} ({overall_daily_percent:.2f}%)")

with col2:
    st.metric("Total Gain / Loss", format_currency(total_gain_loss), delta=f"{overall_return_percent:.2f}%")

with col3:
    st.metric("Total Cost Basis", format_currency(portfolio_details[portfolio_details['Shares'] > 0]['Cost Basis'].sum()))

with col4:
    st.markdown("**Portfolio Weighting**")
    holdings_df = portfolio_details[portfolio_details['Shares'] > 0].copy()
    if not holdings_df.empty and total_value > 0:
        holdings_df.loc[:, 'Weight'] = holdings_df['Current Value'] / total_value
        # show pie chart + small table
        try:
            import plotly.express as px
            fig_pie = px.pie(
                holdings_df,
                names='Ticker',
                values='Current Value',
                title='Holdings by Current Value',
                hole=0.35
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception:
            # fallback if plotly not available
            st.write("Install plotly for pie chart visualization.")
        st.dataframe(holdings_df[['Ticker','Current Value','Weight']].sort_values(by='Current Value', ascending=False).reset_index(drop=True), hide_index=True, width='stretch')
    else:
        st.info("No active holdings to show weights.")

st.markdown("---")

# --- Detailed holdings ---
st.subheader("3 — Detailed Holdings")
display_df = portfolio_details.copy()
if total_value > 0:
    display_df.loc[:, 'Weight'] = np.where(display_df['Shares'] > 0, display_df['Current Value'] / total_value, 0.0)
else:
    display_df.loc[:, 'Weight'] = 0.0

col_order = ['Ticker', 'Shares', 'Avg Cost', 'Current Price', 'Previous Close',
             'Cost Basis', 'Current Value', 'Daily Change ₹', 'Daily Change %', 'Total Gain/Loss', 'Total Return %', 'Weight']
display_df = display_df[[c for c in col_order if c in display_df.columns]]

# Add an expander with another pie + full table for better UX
with st.expander("Show weight chart + full holdings table", expanded=False):
    holdings_df_all = display_df[display_df['Shares'] > 0].copy()
    if not holdings_df_all.empty:
        try:
            import plotly.express as px
            fig_pie2 = px.pie(
                holdings_df_all,
                names='Ticker',
                values='Current Value',
                title='Holdings Weight (detailed)',
                hole=0.35
            )
            fig_pie2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie2, use_container_width=True)
        except Exception:
            st.write("Install plotly for pie chart visualization.")

    st.dataframe(display_df.sort_values(by='Current Value', ascending=False).reset_index(drop=True), hide_index=True, width='stretch')

st.markdown("---")

# --- Historical chart ---
st.subheader(f"4 — Historical Price Trends ({selected_timeframe})")
if history is not None and not history.empty:
    try:
        history_normalized = history.div(history.iloc[0]).mul(100)
        import plotly.express as px
        fig = px.line(history_normalized, title="Normalized Price Performance (Index = 100)", labels={"value": "Normalized (100)", "index": "Date"})
        fig.update_layout(legend_title_text='Asset', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render historical chart: {e}")
else:
    st.info("Historical chart not available for selected tickers/timeframe.")

st.markdown("---")

# --- Footer / last updated / export ---
st.caption("Data source: Yahoo Finance (via yfinance). Prices may be delayed a few minutes during market hours.")
st.write(f"Last updated: {st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S %Z')}")

download_df = display_df.copy()
to_download = io.StringIO()
download_df.to_csv(to_download, index=False)
st.download_button("Download current holdings (CSV)", data=to_download.getvalue(), file_name="holdings_snapshot.csv", mime="text/csv")

missing_list = display_df[display_df['Current Price'].isna()]['Ticker'].tolist()
if missing_list:
    st.warning(f"No price data for: {', '.join(missing_list)}. Check ticker format (use .NS for NSE) or if the asset is not available on Yahoo Finance.")
