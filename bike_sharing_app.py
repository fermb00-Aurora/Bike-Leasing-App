# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load the Dataset (Part I: Exploratory Data Analysis)
# Assuming the dataset 'hour.csv' is in the same directory
bike_data = pd.read_csv('hour.csv')

# Exploratory Data Analysis (Part I: Exploratory Data Analysis)
# Performing a comprehensive analysis of the dataset
st.title('Bike Sharing Service - Exploratory Data Analysis')

# Displaying basic information about the dataset
st.header('Dataset Overview')
st.write(bike_data.head())

# Checking for missing values
st.subheader('Missing Values in Dataset')
st.write(bike_data.isnull().sum())

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write(bike_data.describe())

# Visualizing distributions of key features
st.subheader('Distributions of Numerical Features')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(bike_data['temp'], ax=axs[0, 0], kde=True)
axs[0, 0].set_title('Temperature Distribution')
sns.histplot(bike_data['atemp'], ax=axs[0, 1], kde=True)
axs[0, 1].set_title('Feeling Temperature Distribution')
sns.histplot(bike_data['hum'], ax=axs[1, 0], kde=True)
axs[1, 0].set_title('Humidity Distribution')
sns.histplot(bike_data['windspeed'], ax=axs[1, 1], kde=True)
axs[1, 1].set_title('Windspeed Distribution')
st.pyplot(fig)

# Feature Engineering (Part I: Exploratory Data Analysis)
# Adding new features derived from the existing date column
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
bike_data['year'] = bike_data['dteday'].dt.year
bike_data['month'] = bike_data['dteday'].dt.month
bike_data['day'] = bike_data['dteday'].dt.day
bike_data['dayofweek'] = bike_data['dteday'].dt.dayofweek
bike_data['is_holiday_or_weekend'] = ((bike_data['holiday'] == 1) | (bike_data['workingday'] == 0)).astype(int)

# Correlation Analysis
st.subheader('Correlation Analysis')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(bike_data.corr(), annot=True, cmap='coolwarm', ax=ax)
plt.title('Feature Correlation Heatmap')
st.pyplot(fig)

# Visualizing bike rentals by different features
st.subheader('Bike Rentals by Hour of the Day')
plt.figure(figsize=(10, 6))
sns.boxplot(x='hr', y='cnt', data=bike_data)
plt.title('Bike Rentals by Hour')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Count of Bike Rentals')
st.pyplot(plt)

st.subheader('Bike Rentals by Season')
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='cnt', data=bike_data)
plt.title('Bike Rentals by Season')
plt.xlabel('Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)')
plt.ylabel('Count of Bike Rentals')
st.pyplot(plt)

# Selecting Features and Target (Part II: Prediction Model)
# Features used for training and prediction
target = 'cnt'
features = [
    'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
    'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'is_holiday_or_weekend'
]

# Splitting Features and Target (Part II: Prediction Model)
X = bike_data[features]
y = bike_data[target]

# Scaling Numerical Features (Part II: Prediction Model)
# Using StandardScaler to scale the numerical features for better model performance
scaler = StandardScaler()
numerical_features = ['temp', 'atemp', 'hum', 'windspeed']
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train Random Forest Model (Part II: Prediction Model)
# Training a Random Forest Regressor to predict bike rentals
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the Model and Scaler (Part II: Prediction Model)
# Saving the trained model and scaler for later use in the Streamlit app
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Streamlit App: Title and Sidebar for User Input (Part III: Streamlit Dashboard)
st.title('Bike Sharing Analysis and Prediction Tool')

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
# Load the saved model and scaler to make predictions
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Scale Numerical Features for Input Data (Part III: Streamlit Dashboard)
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make Prediction (Part III: Streamlit Dashboard)
# Use the trained model to make a prediction based on the user input
prediction = rf_model.predict(input_data)[0]

# Display the Prediction (Part III: Streamlit Dashboard)
st.subheader('Predicted Number of Bike Rentals')
st.write(int(prediction))

# Deployment Instructions (Part III: Streamlit Dashboard)
st.sidebar.title('Deployment Instructions')
st.sidebar.write("""
1. **Install Dependencies**: Make sure you have Python and Streamlit installed. You can install required libraries using:
   ```
   pip install -r requirements.txt
   ```

2. **Save Script**: Save this script as `bike_sharing_app.py`.

3. **Prepare Dataset**: Place `hour.csv` in the same directory as `bike_sharing_app.py`.

4. **Run the App**: Use the following command to run the Streamlit app:
   ```
   streamlit run bike_sharing_app.py
   ```

5. **Access the App**: Once the app is running, you can access it in your web browser at `http://localhost:8501`.

6. **Docker Deployment (Optional)**:
   - Create a `Dockerfile` to containerize the app:
     ```Dockerfile
     FROM python:3.8
     WORKDIR /app
     COPY . /app
     RUN pip install -r requirements.txt
     EXPOSE 8501
     ENTRYPOINT ["streamlit", "run"]
     CMD ["bike_sharing_app.py"]
     ```
   - Build and run the Docker container:
     ```
     docker build -t bike_sharing_app .
     docker run -p 8501:8501 bike_sharing_app
     ```
""")
