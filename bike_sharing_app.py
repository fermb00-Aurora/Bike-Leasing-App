# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go

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
st.set_page_config(page_title='Bike Leasing Company Dashboard', page_icon='🚲', layout='wide', initial_sidebar_state='expanded')
st.title('Bike Leasing Company - Data Analysis & Insights')

# Applying Dark Themed Styles
plt.style.use('dark_background')
sns.set(style="darkgrid")

# Dataset Overview - Adding filters and control options
st.sidebar.header('Filter Options')
selected_date = st.sidebar.date_input('Select Analysis Date', pd.to_datetime('2011-12-01'))
time_range = st.sidebar.slider('Business Hours (Hour Range)', 6, 23, (9, 18))
selected_season = st.sidebar.multiselect('Customer Demand Seasonality', options=[1, 2, 3, 4], default=[1, 2, 3, 4], format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1])
weather_condition = st.sidebar.multiselect('Weather Conditions Impact', options=[1, 2, 3, 4], default=[1, 2, 3, 4], format_func=lambda x: ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain/Snow'][x-1])

# Filtering the dataset based on user inputs
filtered_data = bike_data[(bike_data['season'].isin(selected_season)) & (bike_data['weathersit'].isin(weather_condition))]
filtered_data = filtered_data[(filtered_data['hr'] >= time_range[0]) & (filtered_data['hr'] <= time_range[1])]

# Tabs for better organization
tabs = st.tabs(['Data Overview', 'Exploratory Analysis', 'Feature Engineering', 'Correlation Analysis', 'Prediction Model'])

# Data Overview Tab
with tabs[0]:
    st.header('Dataset Overview')
    st.write(filtered_data.head())

    # Comprehensive Data Check
    st.subheader('Data Quality Report')
    missing_values = filtered_data.isnull().sum()
    missing_values_percent = (missing_values / len(filtered_data)) * 100
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
    st.write(filtered_data.dtypes)

    st.write('''Each data type represents the format of the data in the respective column:
    - **int64**: Represents integer values, commonly used for categorical or count data.
    - **float64**: Represents floating-point numbers, used for continuous numerical values like temperature or windspeed.
    - **object**: Represents text or string data, commonly used for date or categorical descriptions.
    Understanding the data types helps in determining the appropriate preprocessing steps, such as scaling numerical features or encoding categorical values.''')

# Exploratory Analysis Tab
with tabs[1]:
    st.header('Exploratory Data Analysis')
    st.subheader('Descriptive Statistics for Numeric Columns')
    st.write(filtered_data.describe())

    st.subheader('Column Analysis - Distribution and Insights')
    for column in filtered_data.columns:
        st.markdown(f'### {column} Analysis')
        if filtered_data[column].isnull().sum() > 0:
            st.warning(f'This column contains {filtered_data[column].isnull().sum()} missing values.')
        if filtered_data[column].dtype in [np.float64, np.int64]:
            st.write(f'Descriptive Stats:')
            st.write(filtered_data[column].describe())
            # Distribution Plot
            if filtered_data[column].nunique() > 10:
                fig, ax = plt.subplots()
                sns.histplot(filtered_data[column], kde=True, ax=ax)
                ax.set_title(f'{column} Distribution')
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                sns.countplot(x=filtered_data[column], ax=ax, palette='viridis')
                ax.set_title(f'{column} Count Plot')
                st.pyplot(fig)
        else:
            st.write(f'Unique Values in {column}: {filtered_data[column].unique()}')

    # Additional Graphs for EDA
    st.subheader('Seasonal Bike Rentals')
    fig = px.bar(filtered_data, x='season', y='cnt', color='season', title='Total Bike Rentals by Season', labels={'season': 'Season', 'cnt': 'Bike Rentals'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Bike Rentals by Day of the Week')
    fig = px.bar(filtered_data, x='weekday', y='cnt', color='weekday', title='Total Bike Rentals by Day of the Week', labels={'weekday': 'Day of the Week', 'cnt': 'Bike Rentals'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Bike Rentals by Weather Situation')
    fig = px.histogram(filtered_data, x='weathersit', y='cnt', color='weathersit', title='Bike Rentals by Weather Situation', labels={'weathersit': 'Weather Situation', 'cnt': 'Bike Rentals'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

# Feature Engineering Tab
with tabs[2]:
    st.header('Feature Engineering')
    filtered_data['dteday'] = pd.to_datetime(filtered_data['dteday'])
    filtered_data['year'] = filtered_data['dteday'].dt.year
    filtered_data['month'] = filtered_data['dteday'].dt.month
    filtered_data['day'] = filtered_data['dteday'].dt.day
    filtered_data['dayofweek'] = filtered_data['dteday'].dt.dayofweek
    filtered_data['is_holiday_or_weekend'] = ((filtered_data['holiday'] == 1) | (filtered_data['workingday'] == 0)).astype(int)

    st.subheader('New Features Added')
    st.write('Added `year`, `month`, `day`, `dayofweek`, and `is_holiday_or_weekend` features for further analysis.')

# Correlation Analysis Tab
with tabs[3]:
    st.header('Correlation Analysis')
    st.subheader('Interactive Correlation Heatmap')
    corr_matrix = filtered_data.corr()
    fig = px.imshow(corr_matrix, color_continuous_scale='viridis', title='Feature Correlation Heatmap', aspect='auto')
    fig.update_traces(hovertemplate='%{x}: %{y} <br>Correlation: %{z:.2f}')
    fig.update_layout(coloraxis_showscale=True, coloraxis_colorbar=dict(title='Correlation'))
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Analysis - Bike Rentals by Feature
    st.subheader('Interactive Visuals: Bike Rentals Analysis')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Bike Rentals by Hour')
        fig = px.box(filtered_data, x='hr', y='cnt', title='Bike Rentals by Hour', color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Bike Rentals by Season')
        fig = px.box(filtered_data, x='season', y='cnt', title='Bike Rentals by Season', color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Bike Rentals by Weather Situation')
    fig = px.box(filtered_data, x='weathersit', y='cnt', title='Bike Rentals by Weather Situation', color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig, use_container_width=True)

    # Data Table for Detailed Information
    st.subheader('Detailed Data View')
    st.dataframe(filtered_data)

# Prediction Model Tab
with tabs[4]:
    st.header('Bike Usage Prediction')
    # Selecting Features and Target (Part II: Prediction Model)
    target = 'cnt'
    features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
        'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'is_holiday_or_weekend'
    ]

    # Splitting Features and Target (Part II: Prediction Model)
    X = filtered_data[features].copy()
    y = filtered_data[target]

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

    # Streamlit App: Sidebar for User Input (Part III: Streamlit Dashboard)
    st.sidebar.header('Input Features for Prediction')
    season = st.sidebar.selectbox('Season', [1, 2, 3, 4], format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1])
    yr = st.sidebar.selectbox('Year', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')
    mnth = st.sidebar.slider('Month', 1, 12, 6)
    hr = st.sidebar.slider('Hour', 0, 23, 12)
    holiday = st.sidebar.selectbox('Holiday Impact', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    weekday = st.sidebar.selectbox('Day of the Week', [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][x])
    workingday = st.sidebar.selectbox('Working Day Indicator', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    weathersit = st.sidebar.selectbox('Weather Conditions', [1, 2, 3, 4], format_func=lambda x: ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain/Snow'][x-1])
    temp = st.sidebar.slider('Average Temperature (Normalized Scale)', 0.0, 1.0, 0.5)
    atemp = st.sidebar.slider('Perceived Temperature (Normalized Scale)', 0.0, 1.0, 0.5)
    hum = st.sidebar.slider('Humidity Level (Normalized Scale)', 0.0, 1.0, 0.5)
    windspeed = st.sidebar.slider('Wind Speed (Normalized Scale)', 0.0, 1.0, 0.3)
    is_holiday_or_weekend = st.sidebar.selectbox('Holiday or Weekend', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    # Creating a DataFrame for the Input Features
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

    # Load Model and Scaler
    try:
        rf_model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model has been trained and saved correctly.")
        st.stop()

    # Scale Numerical Features for Input Data
    try:
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    except ValueError:
        st.error("Input values are out of range for the scaler. Please adjust the inputs.")
        st.stop()

    # Make Prediction
    prediction = rf_model.predict(input_data)[0]

    # Display the Prediction
    st.subheader('Predicted Number of Bike Rentals')
    st.write(f'We predict that there will be **{int(prediction)}** bike rentals for the given conditions.')

    # Improved Visuals for Prediction Result
    st.subheader('Prediction Visualization')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Bike Rentals"},
        gauge={'axis': {'range': [0, max(y)]},
               'bar': {'color': "#2ca02c"},
               'steps': [
                   {'range': [0, max(y) * 0.5], 'color': "lightgray"},
                   {'range': [max(y) * 0.5, max(y)], 'color': "gray"}
               ],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediction}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Option to Save Prediction
    if st.button('Save Prediction'):
        prediction_record = input_data.copy()
        prediction_record['predicted_rentals'] = prediction
        prediction_record.to_csv('saved_predictions.csv', mode='a', header=False, index=False)
        st.success('Prediction saved successfully!')





