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
missing_values_percent = (missing_values / len(bike_data)) * 100
st.write('**Missing Values by Column**')
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_values_percent})
st.write(missing_df)

# Visualize Missing Values
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=missing_values.index, y=missing_values.values, ax=ax, palette='viridis')
ax.set_title('Missing Values by Column')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

st.write('**Data Types Overview**')
st.write(bike_data.dtypes)

st.write('''Each data type represents the format of the data in the respective column:
- **int64**: Represents integer values, commonly used for categorical or count data.
- **float64**: Represents floating-point numbers, used for continuous numerical values like temperature or windspeed.
- **object**: Represents text or string data, commonly used for date or categorical descriptions.
Understanding the data types helps in determining the appropriate preprocessing steps, such as scaling numerical features or encoding categorical values.''')

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
        if bike_data[column].nunique() > 10:
            fig, ax = plt.subplots()
            sns.histplot(bike_data[column], kde=True, ax=ax)
            ax.set_title(f'{column} Distribution')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.countplot(x=bike_data[column], ax=ax, palette='viridis')
            ax.set_title(f'{column} Count Plot')
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
fig.update_layout(coloraxis_showscale=True, coloraxis_colorbar=dict(title='Correlation'))
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
X = bike_data[features].copy()
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
temp = st.sidebar.slider


