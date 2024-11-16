# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# App configuration
st.set_page_config(page_title="🚴‍♂️ Bike Leasing Prediction", layout="centered")

# Load the pre-trained model
@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the pre-trained model (scaler.pkl)."""
    try:
        model = joblib.load("scaler.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# User input function
def user_input():
    st.header("🔍 Enter Bike Leasing Features")

    # User input for key features (customize these based on your dataset)
    season = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"])
    year = st.selectbox("Year", options=[0, 1], help="0 for 2011, 1 for 2012")
    month = st.slider("Month", 1, 12, step=1)
    holiday = st.selectbox("Holiday", options=[0, 1], help="0 for No, 1 for Yes")
    weekday = st.slider("Weekday (0=Sunday, 6=Saturday)", 0, 6)
    workingday = st.selectbox("Working Day", options=[0, 1])
    weather = st.slider("Weather (1=Clear, 4=Heavy Rain)", 1, 4)
    temp = st.slider("Temperature (°C)", 0.0, 40.0, step=0.1)
    humidity = st.slider("Humidity (%)", 0, 100, step=1)
    windspeed = st.slider("Wind Speed (km/h)", 0.0, 50.0, step=0.1)

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        "season": [season],
        "year": [year],
        "month": [month],
        "holiday": [holiday],
        "weekday": [weekday],
        "workingday": [workingday],
        "weather": [weather],
        "temp": [temp],
        "humidity": [humidity],
        "windspeed": [windspeed]
    })

    return input_data

# Main function
def main():
    st.title("🚴‍♂️ Bike Leasing Prediction App")
    st.markdown("Predict the total bike count based on input features.")

    # Get user input
    input_data = user_input()

    # Make prediction
    if st.button("Predict"):
        try:
            # Preprocess input data
            # (Add any necessary preprocessing here based on your model training)
            prediction = model.predict(input_data)[0]
            st.success(f"🚲 Predicted Total Count: {int(prediction)} bikes")
        except Exception as e:
            st.error(f"Prediction error: {e}")
