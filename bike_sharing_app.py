# Import necessary libraries
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ====================== Streamlit Configuration ======================
st.set_page_config(
    page_title='Washington D.C. Bike Sharing Analysis',
    page_icon='🚲',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title and Introduction
st.title('Washington D.C. Bike Sharing Service Analysis')
st.markdown("""
Welcome to the interactive dashboard for the Washington D.C. bike-sharing service analysis. This tool provides insights into the usage patterns of the bike-sharing service and includes recommendations on whether it's a good day to rent a bike based on your input parameters.
""")

# ====================== Load the Dataset ======================
file_path = 'hour.csv'

try:
    bike_data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Dataset 'hour.csv' not found. Please ensure the file is in the correct directory.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Dataset 'hour.csv' is empty. Please provide a valid dataset.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()

# ====================== Load the Pre-trained Model ======================
try:
    # Use pickle_compat for compatibility issues when loading the model (saved as 'scaler.pkl')
    with open('scaler.pkl', 'rb') as model_file:
        model = pd.compat.pickle_compat.load(model_file)
    st.success('Pre-trained model loaded successfully.')

except Exception as e:
    st.error(f"Error loading pre-trained model: {e}")
    st.stop()

# ====================== Data Cleaning & Feature Engineering ======================
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
bike_data['day'] = bike_data['dteday'].dt.day
bike_data['month'] = bike_data['dteday'].dt.month
bike_data['year'] = bike_data['dteday'].dt.year

bike_data['hour_category'] = bike_data['hr'].apply(
    lambda hr: 'Morning' if 6 <= hr < 12 else 'Afternoon' if 12 <= hr < 18 else 'Evening' if 18 <= hr < 24 else 'Night'
)
bike_data = pd.get_dummies(bike_data, columns=['hour_category'], drop_first=True)
bike_data['is_holiday'] = bike_data['holiday'].apply(lambda x: 'Holiday' if x == 1 else 'No Holiday')
bike_data = pd.get_dummies(bike_data, columns=['is_holiday'], drop_first=True)

bike_data['temp_squared'] = bike_data['temp'] ** 2
bike_data['hum_squared'] = bike_data['hum'] ** 2
bike_data['temp_hum_interaction'] = bike_data['temp'] * bike_data['hum']

# ====================== Predictive Modeling and Recommendations ======================
st.header('Recommendations')

season = st.selectbox('Season', [1, 2, 3, 4], format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x])
hr = st.slider('Hour', 0, 23, 12)
holiday = st.selectbox('Holiday', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
workingday = st.selectbox('Working Day', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
weathersit = st.selectbox('Weather Situation', [1, 2, 3, 4], format_func=lambda x: {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain'}[x])
temp = st.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
hum = st.slider('Humidity (normalized)', 0.0, 1.0, 0.5)
windspeed = st.slider('Wind Speed (normalized)', 0.0, 1.0, 0.5)
mnth = st.slider('Month', 1, 12, 6)
weekday = st.slider('Weekday (0=Sunday)', 0, 6, 3)
yr = st.selectbox('Year', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')

# Create a DataFrame for the input features
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
    'windspeed': [windspeed],
})

# Feature engineering for input data
input_data['hour_category'] = input_data['hr'].apply(
    lambda hr: 'Morning' if 6 <= hr < 12 else 'Afternoon' if 12 <= hr < 18 else 'Evening' if 18 <= hr < 24 else 'Night'
)
input_data = pd.get_dummies(input_data, columns=['hour_category'], drop_first=True)
input_data['is_holiday'] = 'Holiday' if holiday == 1 else 'No Holiday'
input_data = pd.get_dummies(input_data, columns=['is_holiday'], drop_first=True)

input_data['temp_squared'] = input_data['temp'] ** 2
input_data['hum_squared'] = input_data['hum'] ** 2
input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

# Align input features with the training data
features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])
missing_cols = set(features) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[features]

# Make predictions
try:
    prediction = model.predict(input_data)
    predicted_count = int(prediction[0])
    st.subheader(f'Predicted Number of Bike Users: **{predicted_count}**')
except Exception as e:
    st.error(f"Error during prediction: {e}")

# Provide recommendation based on the prediction
cnt_mean = bike_data['cnt'].mean()
cnt_std = bike_data['cnt'].std()
if predicted_count >= cnt_mean + cnt_std:
    recommendation = "🌟 It's a great day to rent a bike! High demand expected."
elif predicted_count >= cnt_mean:
    recommendation = "👍 It's a good day to rent a bike."
else:
    recommendation = "🤔 It might not be the best day to rent a bike due to lower demand."

st.write(recommendation)

# ====================== End of Script ======================
