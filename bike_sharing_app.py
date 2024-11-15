# app.py

# Import necessary libraries
import os
import joblib
import warnings

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go

# Streamlit for interactive web applications
import streamlit as st
from streamlit_option_menu import option_menu  # For a modern sidebar menu

# Machine learning libraries
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ====================== Custom Theme and Styling ======================

# Apply custom CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS to enhance the app's appearance
local_css("styles.css")

# ====================== Streamlit Configuration ======================
st.set_page_config(
    page_title='BikeShare Insights',
    page_icon='🚴‍♂️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ====================== Sidebar Navigation Menu ======================
with st.sidebar:
    selected = option_menu(
        menu_title="BikeShare Insights",
        options=["Home", "Data Overview", "Exploratory Analysis", "Recommendations", "About"],
        icons=["house", "table", "bar-chart-line", "lightbulb", "info-circle"],
        menu_icon="bicycle",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#FF4B4B", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#FF4B4B"},
        }
    )

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

# ====================== Data Preprocessing ======================
# Convert 'dteday' to datetime
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])

# ====================== Home Page ======================
if selected == "Home":
    st.title('🚴‍♂️ BikeShare Insights')
    st.markdown("""
    ### Unlock the Potential of Washington D.C.'s Bike-Sharing Data!
    
    Welcome to **BikeShare Insights**, your one-stop platform for interactive data exploration and personalized bike rental recommendations.
    
    **Discover trends, patterns, and get customized suggestions** to enhance your biking experience in the city.
    """)
    # Add an attractive image
    st.image("bike_share_banner.jpg", use_column_width=True)
    
    # Quick summary statistics
    st.markdown("### Quick Stats")
    col1, col2, col3 = st.columns(3)
    total_rentals = bike_data['cnt'].sum()
    average_temp = bike_data['temp'].mean() * 47  # Reverse normalization
    average_windspeed = bike_data['windspeed'].mean() * 67  # Reverse normalization

    col1.metric("Total Rentals", f"{total_rentals:,}")
    col2.metric("Avg Temperature (°C)", f"{average_temp:.2f}")
    col3.metric("Avg Windspeed (km/h)", f"{average_windspeed:.2f}")

# ====================== Data Overview Page ======================
elif selected == "Data Overview":
    st.title('Data Overview')
    st.markdown("""
    ### Dive into the Dataset That Powers Our Insights
    
    Explore the key features and understand the structure of the data.
    """)

    # Show the first few rows of the dataset
    st.subheader('Sample Data')
    st.write(bike_data.head(10))

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

# ====================== Exploratory Analysis Page ======================
elif selected == "Exploratory Analysis":
    st.title('Exploratory Data Analysis')
    st.markdown("""
    ### Uncover Patterns and Trends in Bike-Sharing Usage
    
    Interactive visualizations help you delve deeper into the data.
    """)

    # Interactive plots using Plotly
    st.subheader('Bike Rentals Over Time')
    fig = px.line(bike_data, x='dteday', y='cnt', title='Total Bike Rentals Over Time', labels={'dteday': 'Date', 'cnt': 'Total Rentals'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Average Rentals by Hour')
    avg_hour = bike_data.groupby('hr')['cnt'].mean().reset_index()
    fig = px.bar(avg_hour, x='hr', y='cnt', labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals'}, title='Average Bike Rentals by Hour')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Correlation Heatmap')
    corr = bike_data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(title='Feature Correlation Heatmap', xaxis_nticks=36)
    st.plotly_chart(fig, use_container_width=True)

    # Additional EDA sections can be added here...

# ====================== Recommendations Page ======================
elif selected == "Recommendations":
    st.title('Personalized Bike Rental Recommendations')
    st.markdown("""
    ### Get Insights Tailored to Your Preferences
    
    Adjust the parameters below to receive a recommendation on whether it's a good day for renting a bike.
    """)
    st.write("")

    # Load the pre-trained model and scaler
    try:
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        st.error(f'Error loading pre-trained model and scaler: {e}')
        st.stop()

    # Create a form for user inputs
    with st.form(key='input_form'):
        st.write("#### Customize Your Preferences:")
        col1, col2 = st.columns(2)

        with col1:
            season = st.selectbox(
                'Season',
                [1, 2, 3, 4],
                format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x]
            )
            hr = st.slider('Hour', 0, 23, 12)
            temp = st.slider('Temperature (°C)', -8, 39, 20)
            hum = st.slider('Humidity (%)', 0, 100, 50)
            windspeed = st.slider('Wind Speed (km/h)', 0, 67, 15)

        with col2:
            weathersit = st.selectbox(
                'Weather Situation',
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: 'Clear, Few clouds',
                    2: 'Mist + Cloudy',
                    3: 'Light Snow, Light Rain',
                    4: 'Heavy Rain, Ice Pallets'
                }[x]
            )
            holiday = st.selectbox('Holiday', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            workingday = st.selectbox('Working Day', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            mnth = st.selectbox('Month', list(range(1, 13)), format_func=lambda x: pd.to_datetime(f'2021-{x}-01').strftime('%B'))
            weekday = st.selectbox('Weekday', list(range(0, 7)), format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x])
            yr = st.selectbox('Year', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')

        # Submit button
        submit_button = st.form_submit_button(label='Get Recommendation')

    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'season': [season],
            'yr': [yr],
            'mnth': [mnth],
            'hr': [hr],
            'holiday': [holiday],
            'weekday': [weekday],
            'workingday': [workingday],
            'weathersit': [weathersit],
            'temp': [(temp + 8) / 47],  # Normalize temperature
            'atemp': [(temp + 16) / 66],  # Approximate feels-like temperature normalization
            'hum': [hum / 100],  # Normalize humidity
            'windspeed': [windspeed / 67],  # Normalize windspeed
        })

        # Perform feature engineering
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
        input_data['is_holiday'] = 'Holiday' if holiday == 1 else 'No Holiday'
        input_data = pd.get_dummies(input_data, columns=['is_holiday'], drop_first=True)
        input_data['temp_squared'] = input_data['temp'] ** 2
        input_data['hum_squared'] = input_data['hum'] ** 2
        input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

        # Define the features used during training
        target = 'cnt'
        features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])

        # Ensure the input_data has the same columns as training data
        missing_cols = set(features) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[features]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the pre-trained model
        prediction = best_model.predict(input_data_scaled)
        predicted_count = int(prediction[0])

        # Provide a recommendation based on predicted demand
        cnt_mean = bike_data['cnt'].mean()
        cnt_std = bike_data['cnt'].std()

        if predicted_count >= cnt_mean + cnt_std:
            recommendation = "🌟 It's a fantastic time to rent a bike!"
            emoji = "🚴‍♀️"
        elif predicted_count >= cnt_mean:
            recommendation = "👍 It's a good day for biking."
            emoji = "😊"
        else:
            recommendation = "🤔 You might want to reconsider renting a bike today."
            emoji = "🌧️"

        # Display the recommendation with style
        st.markdown(f"""
        <div style='text-align: center;'>
            <h2>{recommendation}</h2>
            <h1 style='font-size: 80px;'>{emoji}</h1>
            <p><strong>Expected Number of Rentals:</strong> {predicted_count}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show a gauge chart for visual representation
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_count,
            title={'text': "Predicted Rentals"},
            gauge={'axis': {'range': [None, bike_data['cnt'].max()]},
                   'bar': {'color': "#FF4B4B"},
                   'steps': [
                       {'range': [0, cnt_mean], 'color': 'lightgray'},
                       {'range': [cnt_mean, cnt_mean + cnt_std], 'color': 'gray'},
                       {'range': [cnt_mean + cnt_std, bike_data['cnt'].max()], 'color': 'darkgray'}
                   ]}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ====================== About Page ======================
elif selected == "About":
    st.title('About BikeShare Insights')
    st.markdown("""
    ### Empowering Cyclists with Data-Driven Insights
    
    **BikeShare Insights** is developed to help residents and visitors of Washington D.C. make informed decisions about bike rentals. By analyzing historical data, we provide personalized recommendations and highlight trends in bike-sharing usage.
    
    **Features:**
    - Interactive data visualizations
    - Personalized recommendations
    - User-friendly interface
    - Professional design and layout

    **Technologies Used:**
    - Python
    - Streamlit
    - Plotly
    - Machine Learning (Random Forest Regressor)
    - Data Science libraries (Pandas, NumPy, Scikit-learn)

    **Developed by:**
    - *Your Name*
    - [LinkedIn](https://www.linkedin.com)
    - [GitHub](https://www.github.com)

    **Contact Us:**
    If you have any questions or feedback, please reach out at [email@example.com](mailto:email@example.com).
    """)

