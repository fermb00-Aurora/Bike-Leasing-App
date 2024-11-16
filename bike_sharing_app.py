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

# Import load_model from PyCaret
from pycaret.regression import load_model

import pickle

# SIMULATION
 
# Load the pickled model and scaler
pipeline = load_model('final_model')

with tabs[3]:
    st.header('Predictive Modeling')

    # Input features
    # Use more descriptive variable names and provide default values for better UX
    # Streamlit header
    st.header('Bike Rental Prediction')

    # Input for hour of the day (hr)
    hour = st.slider('Select Hour of the Day (hr)', 0, 23, 12)

    # Get the previous hour's cnt value (lag_total_count)
    prev_hour = (hour - 1) % 24  # Ensuring that the hour wraps around (for hour 0, the previous hour is 23)
    prev_cnt = bike_data[bike_data['hr'] == prev_hour]['cnt'].values[0]  # Get cnt value from previous hour

    # Input for temperature and humidity to calculate temp_humidity
    temp = st.slider('Temperature (temp)', 0.0, 1.0, 0.5)
    humidity = st.slider('Humidity (humidity)', 0.0, 1.0, 0.5)
    temp_humidity = temp * humidity  # Calculate temp_humidity as product of temp and humidity

    # Input for working day (workingday)
    workingday = st.selectbox('Working Day', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'lag_total_count': [prev_cnt],  # Previous hour's cnt value (lag_total_count)
        'hr': [hour],
        'temp_humidity': [temp_humidity],
        'workingday': [workingday]
    })

    # Prediction
    if st.button('Predict'):
        # Perform the prediction using the loaded model
        prediction = pipeline.predict(input_data)
        predicted_count = int(prediction[0])  # Get the predicted count
        
        st.subheader(f'Predicted Number of Bike Users: {predicted_count}')



