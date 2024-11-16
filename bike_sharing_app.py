import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import pandas.core.indexes.numeric  # Explicit import to fix the issue

# Streamlit App Configuration
st.set_page_config(
    page_title='Bike Sharing Prediction App',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('🚴‍♂️ Bike Sharing Demand Prediction App')

# Load the model
@st.cache_data(show_spinner=False)
def load_model():
    model_path = "scaler.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is in the correct directory.")
        st.stop()
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
model = load_model()

# Input features
st.header("📊 Input Features for Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    season = st.selectbox("Season", options=[1, 2, 3, 4], help="1: Spring, 2: Summer, 3: Fall, 4: Winter")
    month = st.selectbox("Month", options=list(range(1, 13)))
    hour = st.slider("Hour", min_value=0, max_value=23, value=12)

with col2:
    holiday = st.selectbox("Holiday (0 = No, 1 = Yes)", options=[0, 1])
    weekday = st.selectbox("Weekday (0 = Sunday, 6 = Saturday)", options=list(range(0, 7)))
    workingday = st.selectbox("Working Day (0 = No, 1 = Yes)", options=[0, 1])

with col3:
    temp = st.number_input("Temperature (°C)", value=20.0)
    humidity = st.number_input("Humidity (%)", value=50.0)
    windspeed = st.number_input("Windspeed (m/s)", value=5.0)

# Prepare input data for prediction
input_data = pd.DataFrame([{
    "season": season,
    "month": month,
    "hour": hour,
    "holiday": holiday,
    "weekday": weekday,
    "workingday": workingday,
    "temp": temp,
    "humidity": humidity,
    "windspeed": windspeed
}])

st.subheader("🚀 Prediction")
if st.button("Predict"):
    try:
        # Make a prediction
        prediction = model.predict(input_data)
        st.success(f"Predicted Bike Count: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by Fernando Moreno Borrego")

