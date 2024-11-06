# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Streamlit Configuration
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

# Load the Dataset
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

# Create tabs for different sections
tabs = st.tabs([
    'Data Overview',
    'Data Cleaning & Feature Engineering',
    'Exploratory Data Analysis',
    'Predictive Modeling',
    'Simulator'
])

# Data Overview Tab
with tabs[0]:
    st.header('Data Overview')
    st.write('First, let\'s take a look at the dataset.')

    # Show the first few rows of the dataset
    st.subheader('Raw Data')
    st.write(bike_data.head())

    # Data Summary
    st.subheader('Data Summary')
    st.write(bike_data.describe())

    # Check for missing values
    st.subheader('Missing Values')
    missing_values = bike_data.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Check for duplicate rows
    st.subheader('Duplicate Rows')
    duplicate_rows = bike_data.duplicated().sum()
    st.write(f'Total duplicate rows in the dataset: {duplicate_rows}')

# Data Cleaning & Feature Engineering Tab
with tabs[1]:
    st.header('Data Cleaning & Feature Engineering')

    # Handle missing values (if any)
    st.subheader('Handling Missing Values')
    st.write('No missing values were found in the dataset.')

    # Outlier Detection and Handling
    st.subheader('Outlier Detection and Handling')
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    st.write('Box plots of numerical features to detect outliers.')
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
    axes = axes.flatten()
    for idx, col in enumerate(numerical_features):
        sns.boxplot(y=bike_data[col], ax=axes[idx])
        axes[idx].set_title(col)
    st.pyplot(fig)

    # Handling outliers
    st.write('Outliers are handled using Z-score method.')
    for feature in numerical_features:
        z_scores = np.abs(stats.zscore(bike_data[feature]))
        bike_data = bike_data[(z_scores < 3)]

    # Treatment of text and date features
    st.subheader('Treatment of Text and Date Features')
    bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
    bike_data['day'] = bike_data['dteday'].dt.day
    bike_data['month'] = bike_data['dteday'].dt.month
    bike_data['year'] = bike_data['dteday'].dt.year

    # Generate extra features
    st.subheader('Feature Engineering')

    # Categorize hour into time of day
    def categorize_hour(hr):
        if 6 <= hr < 12:
            return 'Morning'
        elif 12 <= hr < 18:
            return 'Afternoon'
        elif 18 <= hr < 24:
            return 'Evening'
        else:
            return 'Night'
    bike_data['hour_category'] = bike_data['hr'].apply(categorize_hour)
    bike_data = pd.get_dummies(bike_data, columns=['hour_category'], drop_first=True)

    # Encode holiday as binary
    bike_data['is_holiday'] = bike_data['holiday'].apply(lambda x: 'Holiday' if x == 1 else 'No Holiday')
    bike_data = pd.get_dummies(bike_data, columns=['is_holiday'], drop_first=True)

    # Manually create polynomial features
    st.write('Generating polynomial features for temperature and humidity.')
    bike_data['temp_squared'] = bike_data['temp'] ** 2
    bike_data['hum_squared'] = bike_data['hum'] ** 2
    bike_data['temp_hum_interaction'] = bike_data['temp'] * bike_data['hum']
    # You can add more interaction features as needed

# Exploratory Data Analysis Tab
with tabs[2]:
    st.header('Exploratory Data Analysis')

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    corr = bike_data.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm')
    st.pyplot(fig)

    # Insights on features relevant for prediction
    st.write('Features like "temp", "hum", "hr", "season", and "weathersit" show strong correlation with "cnt".')

    # Bike Counts over Hours
    st.subheader('Bike Counts over Hours')
    hour_counts = bike_data.groupby('hr')['cnt'].mean()
    fig = px.line(x=hour_counts.index, y=hour_counts.values, labels={'x': 'Hour of the Day', 'y': 'Average Count'}, title='Average Bike Count per Hour')
    st.plotly_chart(fig)

    # Interactive Feature Exploration
    st.subheader('Interactive Feature Exploration')
    x_axis = st.selectbox('Select X-axis', options=bike_data.columns)
    y_axis = st.selectbox('Select Y-axis', options=bike_data.columns)
    fig = px.scatter(bike_data, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
    st.plotly_chart(fig)

    # Additional Insights
    st.write('From the plots, we observe peak usage during rush hours, indicating that many users are commuting to work.')

# Predictive Modeling Tab
with tabs[3]:
    st.header('Predictive Modeling')

    # Data Preparation
    st.subheader('Data Preparation')
    target = 'cnt'
    features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])
    X = bike_data[features]
    y = bike_data[target]
    st.write('Splitting data into training and testing sets.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    st.write('Scaling features.')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    # Model Training and Evaluation
    st.subheader('Model Training and Evaluation')

    # Linear Regression
    st.write('Training Linear Regression model.')
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    st.write(f'Linear Regression MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}')

    # Ridge Regression with hyperparameter tuning
    st.write('Training Ridge Regression with hyperparameter tuning.')
    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    ridge_model = GridSearchCV(Ridge(), ridge_params, cv=5)
    ridge_model.fit(X_train, y_train)
    best_ridge_model = ridge_model.best_estimator_
    y_pred_ridge = best_ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    st.write(f'Ridge Regression MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}')

    # Random Forest Regression with hyperparameter tuning
    st.write('Training Random Forest Regression with hyperparameter tuning.')
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, None]}
    rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
    rf_model.fit(X_train, y_train)
    best_rf_model = rf_model.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    st.write(f'Random Forest Regression MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')

    # Compare models
    st.subheader('Model Comparison')
    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest'],
        'MSE': [mse_lr, mse_ridge, mse_rf],
        'R2_Score': [r2_lr, r2_ridge, r2_rf]
    })
    st.write(results)

    # Plotting predictions vs reality for the best model
    st.subheader('Predictions vs Actual Values')
    best_model = best_rf_model
    y_pred_best = y_pred_rf
    fig = px.scatter(x=y_test, y=y_pred_best, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title='Actual vs Predicted Bike Counts')
    st.plotly_chart(fig)

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')

# Simulator Tab
with tabs[4]:
    st.header('Bike Usage Prediction Simulator')

    st.write('Use the controls below to input parameters and predict the expected number of bike users.')

    # Load the best model and scaler
    best_model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Input features
    season = st.selectbox('Season', [1, 2, 3, 4], format_func=lambda x: {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}[x])
    hr = st.slider('Hour', 0, 23, 12)
    holiday = st.selectbox('Holiday', [0, 1], format_func=lambda x: 'Yes' if x ==1 else 'No')
    workingday = st.selectbox('Working Day', [0, 1], format_func=lambda x: 'Yes' if x ==1 else 'No')
    weathersit = st.selectbox('Weather Situation', [1, 2, 3, 4], format_func=lambda x: {1:'Clear',2:'Mist',3:'Light Snow/Rain',4:'Heavy Rain'}[x])
    temp = st.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
    hum = st.slider('Humidity (normalized)', 0.0, 1.0, 0.5)
    windspeed = st.slider('Wind Speed (normalized)', 0.0, 1.0, 0.5)
    month = st.slider('Month', 1, 12, 6)
    weekday = st.slider('Weekday', 0, 6, 3)

    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'season': [season],
        'yr': [0],  # Assuming year 2011
        'mnth': [month],
        'hr': [hr],
        'holiday': [holiday],
        'weekday': [weekday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [temp],  # Assuming atemp is similar to temp
        'hum': [hum],
        'windspeed': [windspeed],
        'day': [15],  # Assuming mid-month
        'month': [month],
        'year': [0],  # Assuming year 2011
    })

    # Perform the same feature engineering as before
    # Categorize hour
    def categorize_hour(hr):
        if 6 <= hr < 12:
            return 'Morning'
        elif 12 <= hr < 18:
            return 'Afternoon'
        elif 18 <= hr < 24:
            return 'Evening'
        else:
            return 'Night'
    input_data['hour_category'] = input_data['hr'].apply(categorize_hour)
    input_data = pd.get_dummies(input_data, columns=['hour_category'], drop_first=True)

    # Encode holiday
    input_data['is_holiday'] = 'No Holiday' if holiday == 0 else 'Holiday'
    input_data = pd.get_dummies(input_data, columns=['is_holiday'], drop_first=True)

    # Manually create polynomial features
    input_data['temp_squared'] = input_data['temp'] ** 2
    input_data['hum_squared'] = input_data['hum'] ** 2
    input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

    # Ensure the input_data has the same columns as training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = best_model.predict(input_data_scaled)
    st.subheader(f'Predicted Number of Bike Users: {int(prediction[0])}')


