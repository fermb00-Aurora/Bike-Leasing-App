# Import necessary libraries
import os
import joblib
import warnings
import sys

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for interactive web applications
import streamlit as st

# PDF generation
from fpdf import FPDF

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error
)
import scipy.stats as stats

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
Welcome to the interactive dashboard for the Washington D.C. bike-sharing service analysis. This tool provides insights into the usage patterns of the bike-sharing service and includes a predictive model to estimate the number of users on an hourly basis.
""")

# ====================== Load the Dataset ======================
file_path = 'hour.csv'

# Try to load the dataset, handle exceptions gracefully
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

# ====================== Feature Engineering ======================
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
bike_data['day'] = bike_data['dteday'].dt.day
bike_data['month'] = bike_data['dteday'].dt.month
bike_data['year'] = bike_data['dteday'].dt.year

# Create cyclical features for the hour
bike_data['hr_sin'] = np.sin(2 * np.pi * bike_data['hr'] / 24)
bike_data['hr_cos'] = np.cos(2 * np.pi * bike_data['hr'] / 24)

# Polynomial features
bike_data['temp_squared'] = bike_data['temp'] ** 2
bike_data['hum_squared'] = bike_data['hum'] ** 2
bike_data['temp_hum_interaction'] = bike_data['temp'] * bike_data['hum']

# One-hot encoding for categorical features
bike_data = pd.get_dummies(bike_data, columns=['season', 'weathersit', 'is_holiday'], drop_first=True)

# Define the target variable and features
target = 'cnt'
features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])

# Separate features (X) and target (y)
X = bike_data[features]
y = bike_data[target]

# ====================== Load the Pre-trained Model ======================
try:
    pipeline = joblib.load('scaler.pkl')
    st.success('Pre-trained model loaded successfully.')
except FileNotFoundError:
    st.error("Model file 'scaler.pkl' not found. Please ensure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading pre-trained model: {e}")
    st.stop()

# ====================== Model Evaluation ======================
st.header('Model Evaluation')

# Split the data into a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions using the pre-trained model
try:
    y_pred = pipeline.predict(X_test)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# Display performance metrics
st.write(f'**Model Performance:**')
st.write(f'- Mean Squared Error (MSE): {mse:.2f}')
st.write(f'- Root Mean Squared Error (RMSE): {rmse:.2f}')
st.write(f'- Mean Absolute Error (MAE): {mae:.2f}')
st.write(f'- R² Score: {r2:.2f}')
st.write(f'- Explained Variance Score: {explained_variance:.2f}')
st.write(f'- Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# ====================== Simulator Tab ======================
st.header('Bike Usage Prediction Simulator')

# Input features for the simulator
season = st.selectbox('Season', [1, 2, 3, 4])
hr = st.slider('Hour', 0, 23, 12)
holiday = st.selectbox('Holiday', [0, 1])
workingday = st.selectbox('Working Day', [0, 1])
weathersit = st.selectbox('Weather Situation', [1, 2, 3, 4])
temp = st.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
hum = st.slider('Humidity (normalized)', 0.0, 1.0, 0.5)
windspeed = st.slider('Wind Speed (normalized)', 0.0, 1.0, 0.5)
month = st.slider('Month', 1, 12, 6)
weekday = st.slider('Weekday (0=Sunday)', 0, 6, 3)

# Create a DataFrame for input features
input_data = pd.DataFrame({
    'season': [season],
    'hr': [hr],
    'holiday': [holiday],
    'workingday': [workingday],
    'weathersit': [weathersit],
    'temp': [temp],
    'hum': [hum],
    'windspeed': [windspeed],
    'month': [month],
    'weekday': [weekday]
})

# Feature engineering for input data
input_data['hr_sin'] = np.sin(2 * np.pi * input_data['hr'] / 24)
input_data['hr_cos'] = np.cos(2 * np.pi * input_data['hr'] / 24)
input_data['temp_squared'] = input_data['temp'] ** 2
input_data['hum_squared'] = input_data['hum'] ** 2
input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

# Predict using the loaded model
try:
    prediction = pipeline.predict(input_data)
    predicted_count = int(prediction[0])
    st.subheader(f'Predicted Number of Bike Users: **{predicted_count}**')
except Exception as e:
    st.error(f"Error during prediction: {e}")
