# Import necessary libraries
import os
import joblib
import warnings

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for interactive web applications
import streamlit as st

# Machine learning libraries
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore")

# ====================== Streamlit Configuration ======================
st.set_page_config(
    page_title='Washington D.C. Bike Sharing Analysis',
    page_icon='🚲',
    layout='wide'
)

# Title and Introduction
st.title('Washington D.C. Bike Sharing Service Analysis')
st.markdown("""
This tool provides insights into the usage patterns of the bike-sharing service and includes a predictive model to estimate the number of users based on input parameters.
""")

# ====================== Load Dataset ======================
file_path = 'hour.csv'
try:
    bike_data = pd.read_csv(file_path)
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    st.stop()

# ====================== Load Pre-trained Scaler ======================
try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler loaded successfully from 'scaler.pkl'.")
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# ====================== Data Preparation ======================
features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])
X = bike_data[features]
y = bike_data['cnt']

# Split the data (for evaluation, not for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
try:
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    st.error(f"Error during feature scaling: {e}")
    st.stop()

# ====================== Load Pre-trained Model ======================
try:
    model = joblib.load('best_model.pkl')
    st.success("Model loaded successfully from 'best_model.pkl'.")
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ====================== Model Evaluation ======================
st.header("Model Evaluation")

# Make predictions
try:
    y_pred = model.predict(X_test_scaled)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display metrics
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

# ====================== Simulator Tab ======================
st.header("Bike Usage Prediction Simulator")

# User input
season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])
hr = st.slider("Hour", 0, 23, 12)
holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4], format_func=lambda x: {1: "Clear", 2: "Mist", 3: "Light Snow/Rain", 4: "Heavy Rain"}[x])
temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
windspeed = st.slider("Wind Speed (normalized)", 0.0, 1.0, 0.5)
mnth = st.slider("Month", 1, 12, 6)
weekday = st.slider("Weekday (0=Sunday)", 0, 6, 3)
yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")

# Create input data for prediction
input_data = pd.DataFrame({
    'season': [season],
    'yr': [yr],
    'mnth': [mnth],
    'hr': [hr],
    'holiday': [holiday],
    'weekday': [weekday],
    'workingday': [workingday],
    'weathersit': [weathersit],
    'temp': [temp],
    'atemp': [temp],
    'hum': [hum],
    'windspeed': [windspeed]
})

# Feature scaling
try:
    input_data_scaled = scaler.transform(input_data)
    predicted_count = model.predict(input_data_scaled)[0]
    st.subheader(f"Predicted Number of Bike Users: {int(predicted_count)}")
except Exception as e:
    st.error(f"Error during prediction: {e}")

