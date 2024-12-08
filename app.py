# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the LSTM model
model = load_model("lstm_stock_price_model.keras")

@st.cache_data
def load_data():
    data = pd.read_csv("market_data.csv")
    data = data.sort_values("time_record")
    return data


# Function to predict stock price after a specified number of minutes
def predict_price_after_minutes(model, prices_scaled, seq_length, scaler, future_minutes):
    current_sequence = prices_scaled[-seq_length:].reshape(1, seq_length, 1)
    for _ in range(future_minutes):
        next_scaled = model.predict(current_sequence)[0]
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_scaled]], axis=1)
    final_price = scaler.inverse_transform([[next_scaled[0]]])
    return final_price[0][0]

# Streamlit UI
st.title("Stock Price Prediction with LSTM")

# Load and display the market data
market_data = load_data()
st.write("Latest Market Data", market_data.tail())

# Prepare data for prediction
prices = market_data["close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)
seq_length = 60

# User input for minutes into the future
future_minutes = st.number_input("Enter the number of minutes into the future to predict:", min_value=1, max_value=1000, value=10)

# Predict and display the result
if st.button("Predict"):
    with st.spinner("Predicting... Please wait."):
        predicted_price = predict_price_after_minutes(model, prices_scaled, seq_length, scaler, future_minutes)
    st.success(f"Predicted Close Price after {future_minutes} minutes: {predicted_price:.2f}")
