import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Stock Predictor AI")

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Ensure correct date parsing and sorting
    df = pd.read_csv("stock_data_final.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    return df

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

try:
    df = load_data()
    model = load_model()

    # Predictors must match EXACTLY what you used in training
    predictors = [
        "RSI", "MACD", "MACD_Signal", 
        "Dist_SMA_50", "Dist_SMA_200", 
        "Ret_1d", "Ret_5d", "Day_Range"
    ]

    # --- SIDEBAR ---
    st.sidebar.title("🔍 Configuration")
    ticker_list = df['Ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select Stock", ticker_list)

    # Filter data for selected ticker
    stock_data = df[df['Ticker'] == selected_ticker].copy()
    latest_data = stock_data.iloc[-1] # The most recent day

    # --- MAIN PAGE ---
    st.title(f"📈 AI Analysis: {selected_ticker}")

    # 1. SHOW CHART (Simple Line Chart)
    st.subheader("Price History (Close Price)")

    # We create a simple DataFrame with Date as index for the chart
    chart_data = stock_data[['Date', 'Close']].set_index('Date')
    st.line_chart(chart_data)

    # 2. AI PREDICTION SECTION
    st.markdown("---")
    st.subheader("🤖 Artificial Intelligence Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Based on Weekly Trend Strategy")

    with col2:
        # Extract features for prediction
        input_data = pd.DataFrame([latest_data[predictors]])

        # GET PREDICTION
        prediction_prob = model.predict_proba(input_data)[0][1] # Probability of UP
        prediction_class = (prediction_prob >= 0.55).astype(int) # Using our 0.55 threshold

        # DISPLAY RESULT
        if prediction_class == 1:
            st.success(f"### 🟢 BUY SIGNAL")
            st.write(f"Confidence: **{prediction_prob:.2%}**")
            st.caption("Model expects price to rise in 5 days.")
        else:
            st.error(f"### 🔴 NO TRADE")
            st.write(f"Confidence: **{prediction_prob:.2%}**")
            st.caption("Risk too high. Stay in Cash.")

    with col3:
        # Show the "Why" (Key Metrics)
        st.write("### Key Indicators")
        st.metric("RSI (Momentum)", f"{latest_data['RSI']:.2f}")
        st.metric("Trend (vs 50 SMA)", f"{latest_data['Dist_SMA_50']:.3f}x")
        st.metric("Close Price", f"₹{latest_data['Close']:.2f}")

except Exception as e:
    st.error(f"System Error: {e}")
    st.warning("Did you run 'project_pipeline.py' and save the model first?")
