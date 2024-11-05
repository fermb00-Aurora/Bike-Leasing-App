# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import plotly.express as px

# Load the Dataset (Part I: Exploratory Data Analysis)
# Assuming the dataset 'hour.csv' is in the same directory
file_path = os.path.join(os.path.dirname(__file__), 'hour.csv')
bike_data = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA) - Professional Analysis
st.set_page_config(page_title='Bike Company Dashboard', page_icon='🚲', layout='wide', initial_sidebar_state='expanded')
st.title('Bike Sharing Company - Data Analysis & Insights')

# Applying Dark Themed Styles
plt.style.use('dark_background')
sns.set(style="darkgrid")

# Dataset Overview
st.header('Dataset Overview')
st.write(bike_data.head())

# Comprehensive Data Check
st.subheader('Data Quality Report')
missing_values = bike_data.isnull().sum()
st.write('**Missing Values by Column**')
st.write(missing_values)

st.write('**Data Types Overview**')
st.write(bike_data.dtypes)

# Normalization Check - Summary Stats
st.subheader('Descriptive Statistics for Numeric Columns')
st.write(bike_data.describe())

# Column-by-Column EDA with Detailed Professional Analysis
st.subheader('Column Analysis - Normalization, Missing Values, and Distribution')
for column in bike_data.columns:
    st.markdown(f'### {column} Analysis')
    if bike_data[column].isnull().sum() > 0:
        st.warning(f'This column contains {bike_data[column].isnull().sum()} missing values.')
    if bike_data[column].dtype in [np.float64, np.int64]:
        st.write(f'Descriptive Stats:')
        st.write(bike_data[column].describe())
        # Distribution Plot
        fig, ax = plt.subplots()
        sns.histplot(bike_data[column], kde=True, ax=ax)
        ax.set_title(f'{column} Distribution')
        st.pyplot(fig)
    else:
        st.write(f'Unique Values in {column}: {bike_data[column].unique()}')

# Feature Engineering - Professional Approach
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
bike_data['year'] = bike_data['dteday'].dt.year
bike_data['month'] = bike_data['dteday'].dt.month
bike_data['day'] = bike_data['dteday'].dt.day
bike_data['dayofweek'] = bike_data['dteday'].dt.dayofweek
bike_data['is_holiday_or_weekend'] = ((bike_data['holiday'] == 1) | (bike_data['workingday'] == 0)).astype(int)

st.subheader('New Features Added')
st.write('Added `year`, `month`, `day`, `dayofweek`, and `is_holiday_or_weekend` features for further analysis.')

# Correlation Analysis with Interactive Heatmap
st.subheader('Interactive Correlation Heatmap')
corr_matrix = bike_data.corr()
fig = px.imshow(corr_matrix, color_continuous_scale='viridis', title='Feature Correlation Heatmap', aspect='auto')
fig.update_traces(hovertemplate='%{x}: %{y} <br>Correlation: %{z:.2f}')
fig.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# Interactive Analysis - Bike Rentals by Feature
st.subheader('Interactive Visuals: Bike Rentals Analysis')
col1, col2 = st.columns(2)

with col1:
    st.subheader('Bike Rentals by Hour')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='hr', y='cnt', data=bike_data, palette='viridis')
    ax.set_title('Bike Rentals by Hour')
    st.pyplot(fig)

with col2:
    st.subheader('Bike Rentals by Season')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='season', y='cnt', data=bike_data, palette='viridis')
    ax.set_title('Bike Rentals by Season')
    st.pyplot(fig)

st.subheader('Bike Rentals by Weather Situation')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='weathersit', y='cnt', data=bike_data, palette='viridis')
ax.set_title('Bike Rentals by Weather Situation')
st.pyplot(fig)

# Selecting Features and Target (Part II: Prediction Model)
target = 'cnt'
features = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'is_holiday_or_weekend'
]

# Splitting Features and Target (Part II: Prediction Model)
X = bike_data[features]
y = bike_data[target]

# Scaling Numerical Features (Part II: Prediction Model)
scaler = StandardScaler()
numerical_features = ['temp', 'atemp', 'hum', 'windspeed']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train Random Forest Model (Part II: Prediction Model)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the Model and Scaler (Part II: Prediction Model)
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit App: Title and Sidebar for User Input (Part III: Streamlit Dashboard)
st.title('Bike Sharing Analysis and Prediction Tool - Professional Dashboard')

# Sidebar for User Input (Part III: Streamlit Dashboard)
st.sidebar.header('Input Features for Prediction')
season = st.sidebar.selectbox('Season', [1, 2, 3, 4], format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1])
yr = st.sidebar.selectbox('Year', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')
mnth = st.sidebar.slider('Month', 1, 12, 6)
hr = st.sidebar.slider('Hour', 0, 23, 12)
holiday = st.sidebar.selectbox('Holiday', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
weekday = st.sidebar.slider('Weekday (0 = Sunday)', 0, 6, 0)
workingday = st.sidebar.selectbox('Working Day', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
weathersit = st.sidebar.selectbox('Weather Situation', [1, 2, 3, 4], format_func=lambda x: ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain/Snow'][x-1])
temp = st.sidebar.slider('Temperature (Normalized)', 0.0, 1.0, 0.5)
atemp = st.sidebar.slider('Feeling Temperature (Normalized)', 0.0, 1.0, 0.5)
hum = st.sidebar.slider('Humidity (Normalized)', 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider('Windspeed (Normalized)', 0.0, 1.0, 0.2)
is_holiday_or_weekend = st.sidebar.selectbox('Holiday or Weekend', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Creating a DataFrame for the Input Features (Part III: Streamlit Dashboard)
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
    'atemp': [atemp],
    'hum': [hum],
    'windspeed': [windspeed],
    'is_holiday_or_weekend': [is_holiday_or_weekend]
})

# Load Model and Scaler (Part III: Streamlit Dashboard)
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Scale Numerical Features for Input Data (Part III: Streamlit Dashboard)
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make Prediction (Part III: Streamlit Dashboard)
prediction = rf_model.predict(input_data)[0]

# Display the Prediction (Part III: Streamlit Dashboard)
st.subheader('Predicted Number of Bike Rentals')
st.write(f'We predict that there will be **{int(prediction)}** bike rentals for the given conditions.')
