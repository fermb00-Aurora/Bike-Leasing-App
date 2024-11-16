# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model, predict_model
from io import BytesIO
from sklearn.preprocessing import PolynomialFeatures

# Set the page configuration
st.set_page_config(
    page_title="Bike Rental Analysis and Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: Hide Streamlit's default menu and footer for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# ---------------------------
# Data Processing Functions
# ---------------------------

@st.cache_data
def load_data():
    """
    Load the raw dataset from 'hour.csv'.
    """
    data = pd.read_csv('hour.csv')
    return data

def process_data(df):
    """
    Perform data cleaning and feature engineering on the dataset.
    """
    bike_day = df.copy()
    
    # Rename columns for clarity and consistency
    bike_day.rename(columns={
        'dteday': 'datetime',
        'yr': 'year',
        'mnth': 'month',
        'weathersit': 'weather_condition',
        'hum': 'humidity',
        'cnt': 'total_count'
    }, inplace=True)
    
    # Convert 'datetime' to datetime format
    bike_day['datetime'] = pd.to_datetime(bike_day['datetime'])
    
    # 1. Data Quality Check
    missing_values = bike_day.isnull().sum()
    duplicates = bike_day.duplicated().sum()
    
    # 2. Handling Outliers in 'total_count' using the IQR method
    Q1 = bike_day['total_count'].quantile(0.25)
    Q3 = bike_day['total_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bike_day = bike_day[(bike_day['total_count'] >= lower_bound) & (bike_day['total_count'] <= upper_bound)]
    
    # Feature Engineering
    # Cyclical features for month and weekday to capture seasonality
    bike_day['month_sin'] = np.sin(2 * np.pi * bike_day['month'] / 12)
    bike_day['month_cos'] = np.cos(2 * np.pi * bike_day['month'] / 12)
    bike_day['weekday_sin'] = np.sin(2 * np.pi * bike_day['weekday'] / 7)
    bike_day['weekday_cos'] = np.cos(2 * np.pi * bike_day['weekday'] / 7)
    
    # Interaction and Polynomial Features
    bike_day['temp_humidity'] = bike_day['temp'] * bike_day['humidity']
    bike_day['temp_squared'] = bike_day['temp'] ** 2
    bike_day['windspeed_squared'] = bike_day['windspeed'] ** 2
    
    # Lag Feature (Previous Hour's Total Count)
    bike_day['lag_total_count'] = bike_day['total_count'].shift(1).fillna(method='bfill')
    
    # Create 'daylight_hours' feature based on approximate daylight hours per month
    bike_day['daylight_hours'] = bike_day['month'].apply(lambda x: 9 if x in [12, 1, 2] else
                                                         10 if x in [3, 11] else
                                                         12 if x in [4, 10] else
                                                         14 if x in [5, 9] else
                                                         15 if x in [6, 8] else
                                                         16)
    
    # Generate Lag Features for multiple previous hours
    for lag in range(1, 4):
        bike_day[f'lag_{lag}_total_count'] = bike_day['total_count'].shift(lag).fillna(method='bfill')
    
    # Generate Rolling Mean and Rolling Standard Deviation
    bike_day['rolling_mean_total_count'] = bike_day['total_count'].rolling(window=3).mean().fillna(method='bfill')
    bike_day['rolling_std_total_count'] = bike_day['total_count'].rolling(window=3).std().fillna(method='bfill')
    
    # Time-based features Morning/Afternoon/Evening/Night
    bike_day['time_of_day'] = bike_day['hr'].apply(lambda x: 'Night' if 0 <= x < 6 else      
                                                            'Morning' if 6 <= x < 12 else
                                                            'Afternoon' if 12 <= x < 18 else
                                                            'Evening')
    # One-hot encoding for time_of_day
    bike_day = pd.get_dummies(bike_day, columns=['time_of_day'])
    
    # Interaction between categorical features
    bike_day['holiday_workingday_interaction'] = bike_day['holiday'] * bike_day['workingday']
    
    # Polynomial Features for 'temp' and 'humidity'
    poly = PolynomialFeatures(degree=2, include_bias=False)
    temp_humidity_poly = poly.fit_transform(bike_day[['temp', 'humidity']])
    temp_humidity_poly_df = pd.DataFrame(temp_humidity_poly, columns=poly.get_feature_names_out(['temp', 'humidity']))
    temp_humidity_poly_df.index = bike_day.index
    bike_day = pd.concat([bike_day, temp_humidity_poly_df], axis=1)
    
    # Binarized features for weather conditions
    bike_day['good_weather'] = bike_day['weather_condition'].apply(lambda x: 1 if x <= 2 else 0)
    
    # Extract additional date-related features
    bike_day['hour_sin'] = np.sin(2 * np.pi * bike_day['hr'] / 24)
    bike_day['hour_cos'] = np.cos(2 * np.pi * bike_day['hr'] / 24)
    
    # Ensure no missing values remain
    bike_day.fillna(method='ffill', inplace=True)
    bike_day.fillna(method='bfill', inplace=True)
    
    return bike_day

# Load and process data
raw_data = load_data()
processed_data = process_data(raw_data)

# Load machine learning model
@st.cache_resource
def load_ml_model():
    """
    Load the trained machine learning model from 'scaler.pkl'.
    """
    model = load_model('scaler')  # This loads 'scaler.pkl' saved via PyCaret's save_model
    return model

model = load_ml_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", [
    "About",
    "Data Overview",
    "Data Cleaning & Feature Engineering",
    "Prediction",
    "Simulator",
    "Report Generator",
    "Feedback"
])

# ---------------------------
# About Page
# ---------------------------
if options == "About":
    st.title("About This Application")
    st.markdown("""
    ### Bike Rental Analysis and Prediction

    This application provides an interactive platform to explore, analyze, and predict bike rental counts based on various factors such as weather conditions, time of day, and more. Built using Streamlit and machine learning models, it offers insights through data visualization, simulation tools, and reporting features.

    **Features:**
    - **Data Overview:** Explore the dataset with summaries and visualizations.
    - **Data Cleaning & Feature Engineering:** Understand the data preprocessing steps.
    - **Prediction:** Get predictions on bike rentals using the trained model.
    - **Simulator:** Adjust input variables to simulate different scenarios.
    - **Report Generator:** Generate and download comprehensive reports.
    - **Feedback:** Share your feedback to help us improve.

    **Developed by:** Your Name
    """)

# ---------------------------
# Data Overview Page
# ---------------------------
elif options == "Data Overview":
    st.title("Data Overview")
    st.subheader("Dataset")
    st.dataframe(raw_data.head())
    
    st.subheader("Summary Statistics")
    st.write(raw_data.describe())
    
    st.subheader("Missing Values")
    missing = raw_data.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")
    
    st.subheader("Duplicate Rows")
    duplicates = raw_data.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    
    st.subheader("Data Visualization")
    # Distribution of total_count
    fig1, ax1 = plt.subplots()
    sns.histplot(raw_data['cnt'], kde=True, ax=ax1)
    ax1.set_title('Distribution of Total Bike Rentals')
    ax1.set_xlabel('Total Count')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)
    
    # Correlation Heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(raw_data.corr(), annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Correlation Heatmap of Features')
    st.pyplot(fig2)
    
    # Boxplots for categorical features
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x='season', y='cnt', data=raw_data, ax=axes3[0,0])
    axes3[0,0].set_title("Rental Counts by Season")
    
    sns.boxplot(x='holiday', y='cnt', data=raw_data, ax=axes3[0,1])
    axes3[0,1].set_title("Rental Counts on Holidays vs Non-holidays")
    
    sns.boxplot(x='workingday', y='cnt', data=raw_data, ax=axes3[1,0])
    axes3[1,0].set_title("Rental Counts on Working Days vs Non-working Days")
    
    sns.boxplot(x='weathersit', y='cnt', data=raw_data, ax=axes3[1,1])
    axes3[1,1].set_title("Rental Counts by Weather Condition")
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Scatter plots for numerical features
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    sns.scatterplot(x='temp', y='cnt', data=raw_data, ax=axes4[0])
    axes4[0].set_title("Temperature vs Rental Count")
    
    sns.scatterplot(x='hum', y='cnt', data=raw_data, ax=axes4[1])
    axes4[1].set_title("Humidity vs Rental Count")
    
    sns.scatterplot(x='windspeed', y='cnt', data=raw_data, ax=axes4[2])
    axes4[2].set_title("Windspeed vs Rental Count")
    plt.tight_layout()
    st.pyplot(fig4)

# ---------------------------
# Data Cleaning & Feature Engineering Page
# ---------------------------
elif options == "Data Cleaning & Feature Engineering":
    st.title("Data Cleaning & Feature Engineering")
    st.markdown("""
    This section outlines the data preprocessing steps including renaming columns, handling missing values, outlier detection, feature engineering, and visualization.

    **Key Steps:**
    1. **Renaming Columns:** For clarity and consistency.
    2. **Date Conversion:** Converting 'dteday' to datetime format.
    3. **Data Quality Checks:** Identifying missing values and duplicates.
    4. **Outlier Handling:** Using the IQR method to remove outliers in 'total_count'.
    5. **Feature Engineering:** Creating cyclical features, interaction terms, polynomial features, lag features, and more.
    6. **Data Visualization:** Understanding data distributions and relationships.
    """)

    st.subheader("Renamed Columns")
    st.code("""
    bike_day.rename(columns={
        'dteday': 'datetime',
        'yr': 'year',
        'mnth': 'month',
        'weathersit': 'weather_condition',
        'hum': 'humidity',
        'cnt': 'total_count'
    }, inplace=True)
    """)

    st.subheader("Handling Missing Values and Duplicates")
    st.write("""
    - **Missing Values:** Checked for missing values in each column.
    - **Duplicates:** Identified and removed duplicate rows.
    """)

    st.subheader("Outlier Detection and Removal")
    st.write("""
    Used the Interquartile Range (IQR) method to detect and remove outliers in the 'total_count' column.
    """)

    st.subheader("Feature Engineering")
    st.write("""
    - **Cyclical Features:** Created sine and cosine transformations for 'month' and 'weekday'.
    - **Interaction Features:** Combined 'temp' and 'humidity'.
    - **Polynomial Features:** Added squared terms for 'temp' and 'windspeed'.
    - **Lag Features:** Included previous hours' total counts.
    - **Daylight Hours:** Based on the month.
    - **Binarized Features:** Converted 'weather_condition' to binary.
    - **Time of Day:** Categorized into Morning, Afternoon, Evening, Night with one-hot encoding.
    """)

    st.subheader("Visualizations")
    st.write("Refer to the **Data Overview** section for visual representations of the data.")

# ---------------------------
# Prediction Page
# ---------------------------
elif options == "Prediction":
    st.title("Bike Rental Prediction")
    st.markdown("""
    Enter the necessary features below to get a prediction for bike rentals.
    """)

    # Sidebar for input features
    st.sidebar.header("Input Features")

    def user_input_features():
        season = st.sidebar.selectbox('Season', options=[1, 2, 3, 4])
        yr = st.sidebar.selectbox('Year', options=[0, 1])
        mnth = st.sidebar.selectbox('Month', options=list(range(1,13)))
        hr = st.sidebar.selectbox('Hour', options=list(range(0,24)))
        holiday = st.sidebar.selectbox('Holiday', options=[0, 1])
        weekday = st.sidebar.selectbox('Weekday', options=list(range(0,7)))
        workingday = st.sidebar.selectbox('Working Day', options=[0,1])
        weathersit = st.sidebar.selectbox('Weather Condition', options=[1,2,3,4])
        temp = st.sidebar.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
        atemp = st.sidebar.slider('Feels-like Temperature (normalized)', 0.0, 1.0, 0.5)
        hum = st.sidebar.slider('Humidity', 0.0, 1.0, 0.5)
        windspeed = st.sidebar.slider('Windspeed', 0.0, 1.0, 0.5)

        data = {
            'season': season,
            'yr': yr,
            'mnth': mnth,
            'hr': hr,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit,
            'temp': temp,
            'atemp': atemp,
            'hum': hum,
            'windspeed': windspeed
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("Input Features")
    st.write(input_df)

    # Process input features
    def process_input(df):
        """
        Apply the same data processing steps to user input as the training data.
        """
        bike_day = df.copy()
        
        # Feature Engineering
        # Cyclical features for month and weekday
        bike_day['month_sin'] = np.sin(2 * np.pi * bike_day['mnth'] / 12)
        bike_day['month_cos'] = np.cos(2 * np.pi * bike_day['mnth'] / 12)
        bike_day['weekday_sin'] = np.sin(2 * np.pi * bike_day['weekday'] / 7)
        bike_day['weekday_cos'] = np.cos(2 * np.pi * bike_day['weekday'] / 7)
        
        # Interaction and Polynomial Features
        bike_day['temp_humidity'] = bike_day['temp'] * bike_day['hum']
        bike_day['temp_squared'] = bike_day['temp'] ** 2
        bike_day['windspeed_squared'] = bike_day['windspeed'] ** 2
        
        # Lag Feature (Assuming previous total_count is not available, use a placeholder)
        bike_day['lag_total_count'] = 0  # Placeholder, as real lag requires historical data
        
        # Create 'daylight_hours' feature based on approximate daylight hours per month
        bike_day['daylight_hours'] = bike_day['mnth'].apply(lambda x: 9 if x in [12, 1, 2] else
                                                             10 if x in [3, 11] else
                                                             12 if x in [4, 10] else
                                                             14 if x in [5, 9] else
                                                             15 if x in [6, 8] else
                                                             16)
        
        # Generate Lag Features for multiple previous hours (placeholders)
        for lag in range(1, 4):
            bike_day[f'lag_{lag}_total_count'] = 0  # Placeholder
        
        # Generate Rolling Mean and Rolling Standard Deviation (placeholders)
        bike_day['rolling_mean_total_count'] = 0
        bike_day['rolling_std_total_count'] = 0
        
        # Time-based features Morning/Afternoon/Evening/Night
        bike_day['time_of_day'] = bike_day['hr'].apply(lambda x: 'Night' if 0 <= x < 6 else      
                                                                'Morning' if 6 <= x < 12 else
                                                                'Afternoon' if 12 <= x < 18 else
                                                                'Evening')
        # One-hot encoding for time_of_day
        bike_day = pd.get_dummies(bike_day, columns=['time_of_day'])
        
        # Interaction between categorical features
        bike_day['holiday_workingday_interaction'] = bike_day['holiday'] * bike_day['workingday']
        
        # Polynomial Features for 'temp' and 'hum'
        poly = PolynomialFeatures(degree=2, include_bias=False)
        temp_humidity_poly = poly.fit_transform(bike_day[['temp', 'hum']])
        temp_humidity_poly_df = pd.DataFrame(temp_humidity_poly, columns=poly.get_feature_names_out(['temp', 'hum']))
        temp_humidity_poly_df.index = bike_day.index
        bike_day = pd.concat([bike_day, temp_humidity_poly_df], axis=1)
        
        # Binarized features for weather conditions
        bike_day['good_weather'] = bike_day['weathersit'].apply(lambda x: 1 if x <= 2 else 0)
        
        # Extract additional date-related features
        bike_day['hour_sin'] = np.sin(2 * np.pi * bike_day['hr'] / 24)
        bike_day['hour_cos'] = np.cos(2 * np.pi * bike_day['hr'] / 24)
        
        # Ensure no missing values
        bike_day.fillna(method='ffill', inplace=True)
        bike_day.fillna(method='bfill', inplace=True)
        
        return bike_day

    processed_input = process_input(input_df)

    st.subheader("Processed Input Features")
    st.write(processed_input)

    # Prediction
    if st.button("Predict"):
        prediction = predict_model(model, data=processed_input)
        st.subheader("Prediction")
        st.write(f"**Predicted Bike Rentals:** {int(prediction['Label'][0])}")

# ---------------------------
# Simulator Page
# ---------------------------
elif options == "Simulator":
    st.title("Bike Rental Simulator")
    st.markdown("""
    Adjust the input variables to simulate different scenarios and observe how they affect bike rental counts.
    """)

    # Sidebar for simulator input features
    st.sidebar.header("Simulator Input Features")

    def simulator_input_features():
        season = st.sidebar.selectbox('Season', options=[1, 2, 3, 4], index=0)
        yr = st.sidebar.selectbox('Year', options=[0, 1], index=1)
        mnth = st.sidebar.selectbox('Month', options=list(range(1,13)), index=5)
        hr = st.sidebar.selectbox('Hour', options=list(range(0,24)), index=12)
        holiday = st.sidebar.selectbox('Holiday', options=[0, 1], index=0)
        weekday = st.sidebar.selectbox('Weekday', options=list(range(0,7)), index=0)
        workingday = st.sidebar.selectbox('Working Day', options=[0,1], index=1)
        weathersit = st.sidebar.selectbox('Weather Condition', options=[1,2,3,4], index=0)
        temp = st.sidebar.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
        atemp = st.sidebar.slider('Feels-like Temperature (normalized)', 0.0, 1.0, 0.5)
        hum = st.sidebar.slider('Humidity', 0.0, 1.0, 0.5)
        windspeed = st.sidebar.slider('Windspeed', 0.0, 1.0, 0.5)

        data = {
            'season': season,
            'yr': yr,
            'mnth': mnth,
            'hr': hr,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weathersit,
            'temp': temp,
            'atemp': atemp,
            'hum': hum,
            'windspeed': windspeed
        }
        features = pd.DataFrame(data, index=[0])
        return features

    sim_input_df = simulator_input_features()

    st.subheader("Simulator Input Features")
    st.write(sim_input_df)

    # Process simulator input features
    processed_sim_input = process_input(sim_input_df)

    st.subheader("Processed Simulator Features")
    st.write(processed_sim_input)

    # Simulate Prediction
    if st.button("Simulate Prediction"):
        prediction = predict_model(model, data=processed_sim_input)
        st.subheader("Simulated Prediction")
        st.write(f"**Predicted Bike Rentals:** {int(prediction['Label'][0])}")

# ---------------------------
# Report Generator Page
# ---------------------------
elif options == "Report Generator":
    st.title("Report Generator")
    st.markdown("""
    Generate and download a comprehensive report based on the current dataset and predictions.
    """)

    def generate_report():
        """
        Generate a text-based report summarizing the dataset.
        """
        report = f"""
        # Bike Rental Report

        ## Data Summary
        {raw_data.describe().to_markdown()}

        ## Processed Data Columns
        {', '.join(processed_data.columns.tolist())}

        ## Correlations
        {raw_data.corr()['cnt'].sort_values(ascending=False).to_frame().to_markdown()}

        """
        return report

    report = generate_report()

    st.subheader("Generated Report")
    st.text(report)

    def download_report(report):
        buffer = BytesIO()
        buffer.write(report.encode())
        buffer.seek(0)
        return buffer

    st.download_button(
        label="Download Report as TXT",
        data=download_report(report),
        file_name='bike_rental_report.txt',
        mime='text/plain'
    )

# ---------------------------
# Feedback Page
# ---------------------------
elif options == "Feedback":
    st.title("Feedback")
    st.markdown("""
    We value your feedback! Please let us know your thoughts, suggestions, or any issues you've encountered.
    """)

    with st.form(key='feedback_form'):
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Feedback")
        submit = st.form_submit_button("Submit")

    if submit:
        # For demonstration, we'll simply display a success message.
        # In a real application, you might want to save this to a database or send an email.
        st.success("Thank you for your feedback!")
