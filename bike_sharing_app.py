# app.py

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from pycaret.regression import load_model
from sklearn.preprocessing import PolynomialFeatures

# Streamlit App Configuration
st.set_page_config(page_title="Bike Rental Analysis Dashboard", layout="wide")

# Load the Model
@st.cache_resource
def load_bike_model():
    model_path = 'scaler.pkl'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'scaler.pkl' is in the correct directory.")
        st.stop()
    model = load_model('scaler')
    return model

model = load_bike_model()

# Function to load the dataset
@st.cache_data
def load_data():
    data_path = 'hour.csv'
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please upload 'hour.csv'.")
        st.stop()
    df = pd.read_csv(data_path)
    return df

df = load_data()

# Navigation Menu
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Introduction", "Data Overview", "Feature Engineering", "Prediction", "Simulator", "Download Report", "Feedback"])

# Introduction Page
if page_selection == "Introduction":
    st.title("Bike Rental Analysis Dashboard 🚲")
    st.markdown("""
    This dashboard provides an in-depth analysis of bike rental data and offers predictions for future rentals using a machine learning model.
    
    **Key Features:**
    - Data overview and visualization
    - Feature engineering insights
    - Predictions using a pre-trained model
    - Simulator for custom predictions
    - Downloadable PDF report
    """)

# Data Overview Page
elif page_selection == "Data Overview":
    st.header("Data Overview 📊")
    st.dataframe(df.head())
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Check for missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, title="Feature Correlation Heatmap", color_continuous_scale='coolwarm')
    st.plotly_chart(fig_corr)

# Feature Engineering Page
elif page_selection == "Feature Engineering":
    st.header("Feature Engineering 🛠️")
    st.markdown("Demonstrating the feature engineering steps applied in the dataset.")

    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)

    # Interaction feature
    df['temp_humidity'] = df['temp'] * df['hum']

    st.dataframe(df.head())
    st.markdown("Cyclical features and interaction terms have been added.")

# Simulator Page
elif page_selection == "Simulator":
    st.header("Simulator 🚀")
    st.markdown("Simulate different scenarios and observe the predicted bike rentals.")

    # Input features for simulation
    temp_sim = st.slider("Simulated Temperature", 0.0, 1.0, 0.5)
    hum_sim = st.slider("Simulated Humidity", 0.0, 1.0, 0.5)
    wind_sim = st.slider("Simulated Windspeed", 0.0, 1.0, 0.2)

    sim_input = pd.DataFrame([[temp_sim, hum_sim, wind_sim]], columns=['temp', 'hum', 'windspeed'])
    sim_prediction = model.predict(sim_input)[0]
    st.info(f"Simulated Bike Rentals: {int(sim_prediction)}")

# Download Report Page
elif page_selection == "Download Report":
    st.header("Download Report 📄")

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Bike Rental Analysis Report", ln=True, align='C')
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as file:
            st.download_button("Download PDF", data=file, file_name="report.pdf")

# Feedback Page
elif page_selection == "Feedback":
    st.header("Feedback 💬")
    feedback = st.text_area("Your feedback:")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

