"""
Bike Sharing Prediction Dashboard
Author: Fernando Moreno Borrego
Date: 16.11.2024
Description:
A Streamlit application for analyzing and predicting bike sharing demand using a pre-trained model (`scaler.pkl`).
"""

# Import necessary libraries
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Streamlit App Configuration
st.set_page_config(
    page_title='🚴‍♂️ Bike Sharing Prediction Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title and Sidebar Menu
st.title('🚴‍♂️ Bike Sharing Prediction Dashboard')
st.sidebar.header("Navigation Menu")
page_selection = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Overview",
    "Simulator",
    "Feedback"
])

# Function to load the dataset
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Loads the bike sharing dataset."""
    data_path = 'hour.csv'
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please ensure the dataset is in the correct directory.")
        st.stop()
    df = pd.read_csv(data_path)
    return df

# Function to load the pre-trained model (scaler.pkl)
@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the pre-trained model (scaler.pkl)."""
    model_path = 'scaler.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure `scaler.pkl` is in the correct directory.")
        st.stop()
    model = joblib.load(model_path)
    return model

# Load the dataset and model
df = load_data()
model = load_model()

# Introduction Page
if page_selection == "Introduction":
    st.header("📘 Executive Summary")
    st.markdown("""
    **Objective:**  
    This dashboard provides an interactive analysis and prediction platform for bike sharing demand. It uses a pre-trained model to make predictions based on various features.
    
    **Key Features:**
    - Data Overview: Explore the bike sharing dataset with summary statistics and visualizations.
    - Simulator: Input different features to predict bike rental demand using the pre-trained model.
    - Feedback: Provide your valuable suggestions for improving this dashboard.
    """)

# Data Overview Page
elif page_selection == "Data Overview":
    st.header("🔍 Data Overview")

    # Dataset Preview
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(10).style.highlight_max(axis=0))

    # Summary Statistics
    st.subheader("📊 Data Summary")
    st.dataframe(df.describe().T.style.background_gradient(cmap='YlGnBu'))

    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='YlOrBr',
        title='Correlation Heatmap of Features',
        aspect="auto",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Simulator Page
elif page_selection == "Simulator":
    st.header("🚀 Bike Rental Demand Simulator")
    st.markdown("""
    **Predict Bike Rentals:**  
    Input feature values to predict the total bike rentals using the pre-trained model (`scaler.pkl`).
    """)

    # Input Feature Values
    st.subheader("🔍 Enter Feature Values")
    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
        humidity = st.slider('Humidity (normalized)', 0.0, 1.0, 0.5)
        windspeed = st.slider('Windspeed (normalized)', 0.0, 1.0, 0.5)

    with col2:
        month = st.selectbox('Month', list(range(1, 13)))
        hour = st.selectbox('Hour', list(range(0, 24)))
        weekday = st.selectbox('Weekday', list(range(0, 7)))

    # Create input data for prediction
    input_data = pd.DataFrame([{
        'temp': temp,
        'humidity': humidity,
        'windspeed': windspeed,
        'month': month,
        'hour': hour,
        'weekday': weekday
    }])

    # Make prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"🚴‍♂️ Predicted Total Rentals: {int(prediction)}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Feedback Page
elif page_selection == "Feedback":
    st.header("💬 Feedback")
    st.markdown("""
    **We Value Your Feedback:**  
    Help us improve the Bike Sharing Prediction Dashboard by providing your feedback and suggestions.
    """)

    feedback = st.text_area("Provide your feedback here:")
    if st.button("Submit Feedback"):
        if feedback.strip() == "":
            st.warning("Please enter your feedback before submitting.")
        else:
            st.success("Thank you for your feedback!")

else:
    st.error("Page not found.")
