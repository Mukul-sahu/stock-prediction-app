import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Stock Predictor AI")

# --- LOAD DATA & ASSETS ---
@st.cache_data
def load_data():
    df = pd.read_csv("stock_data_final.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    return df

@st.cache_data
def load_metrics():
    # Load the backtest results
    return pd.read_csv("metrics.csv")

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

try:
    df = load_data()
    metrics_df = load_metrics()
    model = load_model()

    # Predictors must match training
    predictors = [
        "RSI", "MACD", "MACD_Signal", 
        "Dist_SMA_50", "Dist_SMA_200", 
        "Ret_1d", "Ret_5d", "Day_Range"
    ]

    # --- SIDEBAR ---
    st.sidebar.title("🔍 Configuration")
    ticker_list = df['Ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select Stock", ticker_list)

    # Filter data
    stock_data = df[df['Ticker'] == selected_ticker].copy()
    latest_data = stock_data.iloc[-1]

    # Filter metrics for this specific stock
    stock_metrics = metrics_df[metrics_df['Ticker'] == selected_ticker]

    # --- MAIN PAGE ---
    st.title(f"📈 AI Analysis: {selected_ticker}")

    # 1. SHOW CHART
    st.subheader("Price History (Close Price)")
    chart_data = stock_data[['Date', 'Close']].set_index('Date')
    st.line_chart(chart_data)

    # 2. AI PREDICTION
    st.markdown("---")
    st.subheader("🤖 Artificial Intelligence Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Strategy: Weekly Trend Prediction (5-Day Horizon)")

    with col2:
        input_data = pd.DataFrame([latest_data[predictors]])
        prediction_prob = model.predict_proba(input_data)[0][1]
        prediction_class = (prediction_prob >= 0.55).astype(int)

        if prediction_class == 1:
            st.success(f"### 🟢 BUY SIGNAL")
            st.write(f"Confidence: **{prediction_prob:.2%}**")
        else:
            st.error(f"### 🔴 NO TRADE")
            st.write(f"Confidence: **{prediction_prob:.2%}**")

    with col3:
        st.write("### Key Indicators")
        st.metric("RSI", f"{latest_data['RSI']:.2f}")
        st.metric("Trend (vs 50 SMA)", f"{latest_data['Dist_SMA_50']:.3f}x")

    # 3. BACKTEST PERFORMANCE (NEW SECTION)
    st.markdown("---")
    st.subheader(f"📊 Historical Accuracy: {selected_ticker}")

    if not stock_metrics.empty:
        m_col1, m_col2, m_col3 = st.columns(3)

        win_rate = stock_metrics['Win_Rate'].values[0]
        strat_return = stock_metrics['Strategy_Return'].values[0]
        bench_return = stock_metrics['Benchmark_Return'].values[0]

        # Color code the Win Rate
        color = "normal"
        if win_rate > 0.5: color = "inverse"

        with m_col1:
            st.metric("Win Rate (Test Data)", f"{win_rate:.1%}")
        with m_col2:
            st.metric("Model Return", f"{strat_return}x")
        with m_col3:
            st.metric("Buy & Hold Return", f"{bench_return}x")

        if strat_return < bench_return:
            st.warning(f"⚠️ Note: For {selected_ticker}, the model historically underperformed 'Buy & Hold'. Use cautiously.")
        else:
            st.success(f"✅ Note: The model outperformed the market for {selected_ticker}!")
    else:
        st.warning("No backtest data available for this ticker.")

except Exception as e:
    st.error(f"System Error: {e}")




