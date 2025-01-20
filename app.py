import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_curve
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

xgb_model = joblib.load("pkl_files/xgboost_model.pkl")
scaler = joblib.load("pkl_files/scaler.pkl")
pca = joblib.load("pkl_files/pca.pkl")
rfe = joblib.load("pkl_files/rfe.pkl")
saved_feature_names = np.load("pkl_files/feature_names.npy", allow_pickle=True)

def investment_strategy(predicted, vix, usgg3m, gold_threshold=1500):
    if predicted == 1:  
        if vix > 30:  # violatility> favor bonds
            return {"stocks": 0.1, "bonds": 0.6, "gold": 0.3}
        else:  # mid volatility be a bit more risky
            return {"stocks": 0.2, "bonds": 0.5, "gold": 0.3}
    else: 
        if usgg3m < 2:  # low bond yields, favor stocks
            return {"stocks": 0.7, "bonds": 0.2, "gold": 0.1}
        else:  # moderate bond yields
            return {"stocks": 0.6, "bonds": 0.3, "gold": 0.1}

# streamlit App
st.title("AI-Driven Investment Bot")
st.write("Get real-time market crash predictions and investment advice!")

st.header("Enter Financial Indicators")
vix = st.number_input("Volatility Index (VIX)", min_value=0.0, max_value=100.0, value=25.0)
usgg3m = st.number_input("US 3-Month Bond Yield (%)", min_value=0.0, max_value=10.0, value=1.5)
ecsorpus = st.number_input("Economic Surprise Index (ECSURPUS)", value=0.0)
mxeu = st.number_input("MSCI Europe Index (MXEU)", value=0.0)

# Create DataFrame
input_data = pd.DataFrame({
    "VIX": [vix],
    "USGG3M": [usgg3m],
    "ECSURPUS": [ecsorpus],
    "MXEU": [mxeu]
})

# Preprocessing
try:
    input_data = input_data.reindex(columns=saved_feature_names, fill_value=0)
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    input_selected = rfe.transform(input_pca)

    prediction_proba = xgb_model.predict_proba(input_selected)[:, 1][0]

    fpr, tpr, thresholds = joblib.load("pkl_files/xgboost_roc_data.pkl")
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold}")
    predicted_class = int(prediction_proba >= optimal_threshold)

    st.write(f"Prediction Probability: {prediction_proba:.4f}")
    st.write(f"Optimal Threshold: {optimal_threshold:.4f}")
    st.write(f"Predicted Class: {predicted_class}")


    if predicted_class == 1:
        st.warning("Market Crash Predicted! Adjust your investments accordingly.")
    else:
        st.success("No Market Crash Predicted. Consider a growth-oriented strategy.")

    strategy = investment_strategy(predicted_class, vix, usgg3m)
    st.write("### Recommended Portfolio Allocation")
    st.write(f"- **Stocks**: {strategy['stocks'] * 100:.1f}%")
    st.write(f"- **Bonds**: {strategy['bonds'] * 100:.1f}%")
    st.write(f"- **Gold**: {strategy['gold'] * 100:.1f}%")

    #simulate portfolio
    initial_investment = st.number_input("Enter Initial Investment ($)", min_value=1000, value=100000)
    portfolio_value = initial_investment
    returns = (
        strategy["stocks"] * vix * 0.0001 +
        strategy["bonds"] * usgg3m * 0.00005 +
        strategy["gold"] * mxeu * 0.00008
    )
    portfolio_value *= (1 + np.clip(returns, -0.01, 0.01))
    st.write(f"Projected Portfolio Value After 1 Day: **${portfolio_value:,.2f}**")

    #plotting growth
    portfolio_growth = [initial_investment]
    for _ in range(10):  # Simulate 10 days
        daily_returns = (
            strategy["stocks"] * vix * 0.0001 +
            strategy["bonds"] * usgg3m * 0.00005 +
            strategy["gold"] * mxeu * 0.00008
        )
        daily_returns = np.clip(daily_returns, -0.01, 0.01)
        portfolio_growth.append(portfolio_growth[-1] * (1 + daily_returns))

    st.write("### Portfolio Growth Over Time")
    st.line_chart(portfolio_growth)

except Exception as e:
    logging.error(f"Error in prediction: {str(e)}", exc_info=True)
    st.error("An error occurred while processing your request. Please check the input values.")
