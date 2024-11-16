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
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ====================== Streamlit Configuration ======================
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

# ====================== Create Tabs for Different Sections ======================
tabs = st.tabs([
    'Data Overview',
    'Data Cleaning & Feature Engineering',
    'Exploratory Data Analysis',
    'Predictive Modeling',
    'Simulator',
    'Download Report',
    'Feedback',
    'About'
])

# ====================== Data Overview Tab ======================
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
    if missing_values.sum() > 0:
        st.write(missing_values[missing_values > 0])
    else:
        st.write("No missing values found.")

    # Check for duplicate rows
    st.subheader('Duplicate Rows')
    duplicate_rows = bike_data.duplicated().sum()
    st.write(f'Total duplicate rows in the dataset: {duplicate_rows}')

# ====================== Data Cleaning & Feature Engineering Tab ======================
with tabs[1]:
    st.header('Data Cleaning & Feature Engineering')

    # Handle missing values (if any)
    st.subheader('Handling Missing Values')
    if missing_values.sum() == 0:
        st.write('No missing values were found in the dataset.')
    else:
        st.write('Missing values detected. Proceeding to handle them.')
        # Implement missing value handling here if necessary

    # Outlier Detection and Handling
    st.subheader('Outlier Detection and Handling')

    # Define numerical features for outlier detection
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

    st.write('Box plots of numerical features to detect outliers.')

    # Create box plots for each numerical feature
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
    axes = axes.flatten()
    for idx, col in enumerate(numerical_features):
        sns.boxplot(y=bike_data[col], ax=axes[idx])
        axes[idx].set_title(col)
    # Hide the last subplot if there are fewer features
    for idx in range(len(numerical_features), len(axes)):
        axes[idx].set_visible(False)
    st.pyplot(fig)

    # Handling outliers using the Z-score method
    st.write('Outliers are handled using the Z-score method (threshold = 3).')

    # Remove outliers from the dataset
    for feature in numerical_features:
        z_scores = np.abs(stats.zscore(bike_data[feature]))
        bike_data = bike_data[(z_scores < 3)]

    # Treatment of text and date features
    st.subheader('Treatment of Text and Date Features')

    # Convert 'dteday' to datetime and extract date components
    bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
    bike_data['day'] = bike_data['dteday'].dt.day
    bike_data['month'] = bike_data['dteday'].dt.month
    bike_data['year'] = bike_data['dteday'].dt.year

    # Feature Engineering
    st.subheader('Feature Engineering')

    st.write('Categorizing hours into time of day and creating polynomial features.')

    # Function to categorize hour into time of day
    def categorize_hour(hr):
        if 6 <= hr < 12:
            return 'Morning'
        elif 12 <= hr < 18:
            return 'Afternoon'
        elif 18 <= hr < 24:
            return 'Evening'
        else:
            return 'Night'

    # Apply the function to create a new feature
    bike_data['hour_category'] = bike_data['hr'].apply(categorize_hour)

    # One-hot encode the 'hour_category' feature
    bike_data = pd.get_dummies(bike_data, columns=['hour_category'], drop_first=True)

    # Encode 'holiday' as a categorical feature
    bike_data['is_holiday'] = bike_data['holiday'].apply(lambda x: 'Holiday' if x == 1 else 'No Holiday')
    bike_data = pd.get_dummies(bike_data, columns=['is_holiday'], drop_first=True)

    # Create polynomial features for 'temp' and 'hum'
    bike_data['temp_squared'] = bike_data['temp'] ** 2
    bike_data['hum_squared'] = bike_data['hum'] ** 2
    bike_data['temp_hum_interaction'] = bike_data['temp'] * bike_data['hum']

# ====================== Exploratory Data Analysis Tab ======================
with tabs[2]:
    st.header('Exploratory Data Analysis')

    # ====================== 1. Distribution of Total Bike Rentals ======================
    st.subheader('1. Distribution of Total Bike Rentals')
    st.write('The histogram shows the overall distribution of total rentals, helping identify skewness, central tendencies, and outliers.')

    fig = px.histogram(bike_data, x='cnt', nbins=50, title='Distribution of Total Bike Rentals')
    st.plotly_chart(fig)

    # Compute statistics
    mean_cnt = bike_data['cnt'].mean()
    median_cnt = bike_data['cnt'].median()
    skewness_cnt = bike_data['cnt'].skew()

    # Display dynamic comments
    st.write(f"**Average Rentals:** {mean_cnt:.2f}")
    st.write(f"**Median Rentals:** {median_cnt:.2f}")
    st.write(f"**Skewness:** {skewness_cnt:.2f}")
    if skewness_cnt > 0:
        st.write('The distribution is right-skewed, indicating a longer tail on the right.')
    elif skewness_cnt < 0:
        st.write('The distribution is left-skewed, indicating a longer tail on the left.')
    else:
        st.write('The distribution is symmetric.')

    # ====================== 2. Correlation Heatmap ======================
    st.subheader('2. Correlation Heatmap')
    st.write('Highlights relationships between features.')

    corr = bike_data.corr()
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm')
    st.pyplot(fig)

    # ====================== 3. Lineplot: Average Bike Rentals by Hour ======================
    st.subheader('3. Average Bike Rentals by Hour')
    st.write('Shows daily rental trends with peaks during commuting hours.')

    hour_counts = bike_data.groupby('hr')['cnt'].mean()
    fig = px.line(
        x=hour_counts.index,
        y=hour_counts.values,
        labels={'x': 'Hour of the Day', 'y': 'Average Count'},
        title='Average Bike Rentals by Hour'
    )
    st.plotly_chart(fig)

    # ====================== 4. Boxplots: Rentals by Season ======================
    st.subheader('4. Rentals by Season')
    st.write('Rentals peak in spring and summer due to favorable weather and decline in winter.')

    fig = px.box(bike_data, x='season', y='cnt', labels={'season': 'Season', 'cnt': 'Total Rentals'}, title='Rentals by Season')
    fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Spring', 'Summer', 'Fall', 'Winter']
    )
    st.plotly_chart(fig)

    # ====================== 5. Boxplots: Rentals by Weekday ======================
    st.subheader('5. Rentals by Weekday')
    st.write('Weekdays show consistent commuting patterns, while weekends exhibit more variability due to leisure activities.')

    fig = px.box(bike_data, x='weekday', y='cnt', labels={'weekday': 'Weekday', 'cnt': 'Total Rentals'}, title='Rentals by Weekday')
    fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4, 5, 6],
        ticktext=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    )
    st.plotly_chart(fig)

    # ====================== 6. Boxplots: Rentals by Weather Condition ======================
    st.subheader('6. Rentals by Weather Condition')
    st.write('Clear weather drives the highest rentals.')

    fig = px.box(bike_data, x='weathersit', y='cnt', labels={'weathersit': 'Weather Situation', 'cnt': 'Total Rentals'}, title='Rentals by Weather Condition')
    fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain']
    )
    st.plotly_chart(fig)

    # ====================== 7. Scatterplot: Temperature vs Total Rentals ======================
    st.subheader('7. Temperature vs Total Rentals')
    st.write('Positive relationship with higher rentals at moderate temperatures.')

    fig = px.scatter(bike_data, x='temp', y='cnt', labels={'temp': 'Temperature (Normalized)', 'cnt': 'Total Rentals'}, title='Temperature vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 8. Scatterplot: Humidity vs Total Rentals ======================
    st.subheader('8. Humidity vs Total Rentals')
    st.write('Rentals decrease as humidity increases.')

    fig = px.scatter(bike_data, x='hum', y='cnt', labels={'hum': 'Humidity (Normalized)', 'cnt': 'Total Rentals'}, title='Humidity vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 9. Scatterplot: Windspeed vs Total Rentals ======================
    st.subheader('9. Windspeed vs Total Rentals')
    st.write('Rentals are highest at low wind speeds and decline as windspeed increases.')

    fig = px.scatter(bike_data, x='windspeed', y='cnt', labels={'windspeed': 'Windspeed (Normalized)', 'cnt': 'Total Rentals'}, title='Windspeed vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 10. Lineplot: Total Rentals Over Time ======================
    st.subheader('10. Total Rentals Over Time')
    st.write('Displays seasonal trends with peaks during summer and dips during winter.')

    # Ensure 'dteday' is in datetime format
    bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])

    fig = px.line(bike_data, x='dteday', y='cnt', labels={'dteday': 'Date', 'cnt': 'Total Rentals'}, title='Total Rentals Over Time')
    st.plotly_chart(fig)

    # ====================== 11. Boxplots: Rentals by Low Traffic and Busy Hours ======================
    st.subheader('11. Rentals by Low Traffic and Busy Hours')
    st.write('Analyzing rentals during low traffic (early morning, late night) vs busy hours (commuting times).')

    # Define low traffic and busy hours
    bike_data['traffic_category'] = bike_data['hr'].apply(lambda x: 'Busy' if (7 <= x <= 9) or (17 <= x <= 19) else 'Low Traffic')

    fig = px.box(bike_data, x='traffic_category', y='cnt', labels={'traffic_category': 'Traffic Category', 'cnt': 'Total Rentals'}, title='Rentals by Traffic Category')
    st.plotly_chart(fig)

    # ====================== 12. Rentals by User Type (Casual vs Registered) ======================
    st.subheader('12. Rentals by User Type (Casual vs Registered)')
    st.write('Comparing rental patterns between casual and registered users.')

    fig = px.line(
        bike_data,
        x='dteday',
        y=['casual', 'registered'],
        labels={'value': 'Number of Users', 'dteday': 'Date', 'variable': 'User Type'},
        title='Rentals by User Type Over Time'
    )
    st.plotly_chart(fig)

    # ====================== 13. Boxplots: Rentals by Holidays Without Weekends ======================
    st.subheader('13. Rentals by Holidays Without Weekends')
    st.write('Rentals are higher on holidays due to increased leisure activities.')

    # Filter out weekends
    weekdays = bike_data[bike_data['weekday'].isin([1, 2, 3, 4, 5])]  # Monday to Friday

    fig = px.box(weekdays, x='holiday', y='cnt', labels={'holiday': 'Holiday', 'cnt': 'Total Rentals'}, title='Rentals on Holidays vs Non-Holidays (Weekdays)')
    fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['No Holiday', 'Holiday']
    )
    st.plotly_chart(fig)

    # ====================== 14. Cyclical Encoding: Hour (Sin and Cosine) ======================
    st.subheader('14. Hourly Rentals with Cyclical Encoding')
    st.write('Visualizing the cyclical nature of hourly rentals.')

    # Create cyclical features
    bike_data['hr_sin'] = np.sin(2 * np.pi * bike_data['hr'] / 24)
    bike_data['hr_cos'] = np.cos(2 * np.pi * bike_data['hr'] / 24)

    fig = px.scatter_3d(
        bike_data, x='hr_sin', y='hr_cos', z='cnt',
        labels={'hr_sin': 'Hour Sin', 'hr_cos': 'Hour Cos', 'cnt': 'Total Rentals'},
        title='Hourly Rentals with Cyclical Encoding'
    )
    st.plotly_chart(fig)

    # ====================== 15. Histograms: Temperature, Humidity, Windspeed ======================
    st.subheader('15. Histograms of Environmental Variables')
    st.write('Displaying the frequency distribution of temperature, humidity, and windspeed.')

    # Create histograms
    fig_temp = px.histogram(bike_data, x='temp', nbins=50, title='Temperature Distribution')
    fig_hum = px.histogram(bike_data, x='hum', nbins=50, title='Humidity Distribution')
    fig_wind = px.histogram(bike_data, x='windspeed', nbins=50, title='Windspeed Distribution')

    st.plotly_chart(fig_temp)
    st.plotly_chart(fig_hum)
    st.plotly_chart(fig_wind)

    # ====================== 16. Scatterplot: Lag Feature vs Total Rentals ======================
    st.subheader('16. Lag Feature vs Total Rentals')
    st.write('Exploring temporal dependency where high rentals in the previous hour often lead to high rentals in the current hour.')

    # Create lag feature
    bike_data['cnt_lag1'] = bike_data['cnt'].shift(1)
    bike_data.dropna(inplace=True)

    fig = px.scatter(bike_data, x='cnt_lag1', y='cnt', labels={'cnt_lag1': 'Previous Hour Rentals', 'cnt': 'Total Rentals'}, title='Lag Feature vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 17. Boxplots: Squared Features (Temp Squared, Windspeed Squared) ======================
    st.subheader('17. Squared Features vs Total Rentals')
    st.write('Analyzing how extreme values in temperature and windspeed affect rentals.')

    # Boxplot for temp_squared
    fig_temp_sq = px.box(bike_data, x='season', y='temp_squared', labels={'season': 'Season', 'temp_squared': 'Temp Squared'}, title='Temp Squared by Season')
    fig_temp_sq.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Spring', 'Summer', 'Fall', 'Winter']
    )
    st.plotly_chart(fig_temp_sq)

    # Boxplot for windspeed_squared
    bike_data['windspeed_squared'] = bike_data['windspeed'] ** 2
    fig_wind_sq = px.box(bike_data, x='season', y='windspeed_squared', labels={'season': 'Season', 'windspeed_squared': 'Windspeed Squared'}, title='Windspeed Squared by Season')
    fig_wind_sq.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Spring', 'Summer', 'Fall', 'Winter']
    )
    st.plotly_chart(fig_wind_sq)

    # ====================== 18. Boxplot: Temp-Humidity Interaction vs Total Rentals ======================
    st.subheader('18. Temp-Humidity Interaction vs Total Rentals')
    st.write('Analyzing how combinations of temperature and humidity influence rental behavior.')

    fig = px.scatter(bike_data, x='temp_hum_interaction', y='cnt', labels={'temp_hum_interaction': 'Temp-Humidity Interaction', 'cnt': 'Total Rentals'}, title='Temp-Humidity Interaction vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 19. Scatterplot: Rolling 3-Hour Average vs Total Rentals ======================
    st.subheader('19. Rolling 3-Hour Average vs Total Rentals')
    st.write('Demonstrating short-term rental trends.')

    bike_data['cnt_roll3'] = bike_data['cnt'].rolling(window=3).mean()
    bike_data.dropna(inplace=True)

    fig = px.scatter(bike_data, x='cnt_roll3', y='cnt', labels={'cnt_roll3': '3-Hour Rolling Average', 'cnt': 'Total Rentals'}, title='Rolling 3-Hour Average vs Total Rentals')
    st.plotly_chart(fig)

    # ====================== 20. Boxplot: Total Rentals by Daylight Hours ======================
    st.subheader('20. Total Rentals by Daylight Hours')
    st.write('Rentals increase with longer daylight hours, reflecting seasonal variations and user behavior.')

    # Assuming 'hr' can represent daylight hours (simplified)
    bike_data['daylight'] = bike_data['hr'].apply(lambda x: 'Daylight' if 6 <= x <= 18 else 'Night')

    fig = px.box(bike_data, x='daylight', y='cnt', labels={'daylight': 'Daylight', 'cnt': 'Total Rentals'}, title='Total Rentals by Daylight Hours')
    st.plotly_chart(fig)

    # ====================== Key Takeaways ======================
    st.header('Key Takeaways')

    st.markdown("""
    - **Critical Graphs**:
      - Distribution of rentals.
      - Correlation heatmap.
      - Scatterplots of temperature, humidity, and windspeed vs rentals.
      - Hourly trends in rentals.
    - **Seasonality and Weather**:
      - Clear seasonal patterns highlight the role of climate and daylight.
      - Weather conditions like clear skies and moderate temperatures are key drivers of rentals.
    - **Temporal and User Behavior**:
      - Hourly, daily, and user-type trends emphasize structured rental patterns tied to commuting and leisure.
    - **Feature Engineering**:
      - Engineered features (e.g., lag, rolling averages, cyclical encoding) reveal valuable temporal dependencies and patterns.
    """)

# ====================== Predictive Modeling Tab ======================
with tabs[3]:
    st.header('Predictive Modeling')

    # Load the pre-trained model and scaler (assuming scaler.pkl contains the pipeline)
    st.subheader('Load Pre-trained Model and Scaler')

    try:
        # Load the scaler (pipeline that includes the model)
        pipeline = joblib.load('scaler.pkl')
        st.success('Pre-trained model and scaler loaded successfully.')
    except FileNotFoundError:
        st.error("File 'scaler.pkl' not found. Please ensure it is in the correct directory.")
        st.stop()
    except ImportError as ie:
        st.error(f'Import Error: {ie}')
        st.stop()
    except Exception as e:
        error_msg = str(e)
        st.error(f'Error loading pre-trained model and scaler: {error_msg}')

        # Check if the error is related to Python version incompatibility
        if 'Pycaret only supports python' in error_msg:
            # URL of the GIF you want to display
            gif_url = "https://media.giphy.com/media/3o6UB4cLhGn9JjdT7y/giphy.gif"  # Replace with your desired GIF URL
            st.image(gif_url, caption='Please downgrade your Python version to 3.11 or below.', use_column_width=True)

        st.stop()

    # Data Preparation
    st.subheader('Data Preparation')

    # Define the target variable and features
    target = 'cnt'
    features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])

    # Separate features (X) and target (y)
    X = bike_data[features]
    y = bike_data[target]

    st.write('Splitting data into training and testing sets.')

    # Split the data into training and testing sets
    # Note: Even though we are not retraining the model, we need a test set to evaluate the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Evaluation
    st.subheader('Model Evaluation')

    # Use the pre-trained pipeline to make predictions on the test set
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        st.error(f'Error during prediction: {e}')
        st.stop()

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'**Loaded Model Performance:**')
    st.write(f'- Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'- R² Score: {r2:.2f}')

    # Plotting predictions vs actual values
    st.subheader('Predictions vs Actual Values')

    # Plot actual vs predicted values
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title='Actual vs Predicted Bike Counts',
        trendline='ols'  # Adds a trendline to visualize correlation
    )
    st.plotly_chart(fig)

# ====================== Simulator Tab ======================
with tabs[4]:
    st.header('Bike Usage Prediction Simulator')

    st.write('Use the controls below to input parameters and predict the expected number of bike users.')

    # Load the pre-trained model and scaler
    try:
        pipeline = joblib.load('scaler.pkl')
    except Exception as e:
        st.error(f'Error loading pre-trained model and scaler: {e}')
        st.stop()

    # Input features
    # Use more descriptive variable names and provide default values for better UX
    season = st.selectbox(
        'Season',
        [1, 2, 3, 4],
        format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x]
    )
    hr = st.slider('Hour', 0, 23, 12)
    holiday = st.selectbox('Holiday', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    workingday = st.selectbox('Working Day', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    weathersit = st.selectbox(
        'Weather Situation',
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: 'Clear',
            2: 'Mist',
            3: 'Light Snow/Rain',
            4: 'Heavy Rain'
        }[x]
    )
    temp = st.slider('Temperature (normalized)', 0.0, 1.0, 0.5)
    hum = st.slider('Humidity (normalized)', 0.0, 1.0, 0.5)
    windspeed = st.slider('Wind Speed (normalized)', 0.0, 1.0, 0.5)
    month = st.slider('Month', 1, 12, 6)
    weekday = st.slider('Weekday (0=Sunday)', 0, 6, 3)

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
        'atemp': [temp],  # Assuming 'atemp' is similar to 'temp'
        'hum': [hum],
        'windspeed': [windspeed],
        'day': [15],  # Assuming mid-month
        'month': [month],
        'year': [0],  # Assuming year 2011
    })

    # Perform the same feature engineering as before
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

    # Apply the function to create a new feature
    input_data['hour_category'] = input_data['hr'].apply(categorize_hour)

    # One-hot encode the 'hour_category' feature
    input_data = pd.get_dummies(input_data, columns=['hour_category'], drop_first=True)

    # Encode 'is_holiday' as a categorical feature
    input_data['is_holiday'] = 'No Holiday' if holiday == 0 else 'Holiday'
    input_data = pd.get_dummies(input_data, columns=['is_holiday'], drop_first=True)

    # Create polynomial features for 'temp' and 'hum'
    input_data['temp_squared'] = input_data['temp'] ** 2
    input_data['hum_squared'] = input_data['hum'] ** 2
    input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

    # Ensure the input_data has the same columns as training data
    missing_cols = set(bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])]

    # Predict using the pre-trained pipeline
    try:
        prediction = pipeline.predict(input_data)
        predicted_count = int(prediction[0])
        st.subheader(f'Predicted Number of Bike Users: **{predicted_count}**')
    except Exception as e:
        st.error(f'Error during prediction: {e}')

# ====================== Download Report Tab ======================
with tabs[5]:
    st.header('📄 Download Report')

    st.markdown("""
    **Generate and Download a Professional PDF Report:**
    Compile your analysis and model evaluation results into a comprehensive and business-oriented PDF report for offline review and sharing with stakeholders.
    """)

    # Button to generate report
    if st.button("Generate Report"):
        with st.spinner("Generating PDF report..."):
            try:
                # Initialize PDF
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)

                # Title Page
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Washington D.C. Bike Sharing Analysis Report", ln=True, align='C')
                pdf.ln(10)

                # Executive Summary
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Executive Summary", ln=True)
                pdf.set_font("Arial", '', 12)
                exec_summary = (
                    "This report provides a comprehensive analysis of the Washington D.C. bike-sharing service, focusing on user behavior, environmental impacts, and temporal trends. "
                    "It offers actionable insights to optimize operations, enhance customer satisfaction, and increase revenue."
                )
                pdf.multi_cell(0, 10, exec_summary)
                pdf.ln(5)

                # Data Overview
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Data Overview", ln=True)
                pdf.set_font("Arial", '', 12)
                total_records = len(bike_data)
                data_overview = (
                    f"- **Total Records:** {total_records:,}\n"
                    "- **Features:** Hourly rental data, user types, weather conditions, and temporal information.\n"
                    "- **Time Period:** Covers bike-sharing data over two years.\n"
                )
                pdf.multi_cell(0, 10, data_overview)
                pdf.ln(5)

                # Key Insights
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Key Insights", ln=True)
                pdf.set_font("Arial", '', 12)
                insights = (
                    "- **Peak Usage Hours:** Rentals peak during morning and evening commute times (7–9 AM, 5–7 PM).\n"
                    "- **Seasonal Trends:** Higher usage observed during spring and summer due to favorable weather.\n"
                    "- **Weather Impact:** Clear weather conditions significantly increase rentals, while adverse conditions decrease them.\n"
                    "- **User Behavior:** Registered users exhibit consistent weekday patterns, while casual users are more active on weekends and holidays.\n"
                    "- **Environmental Factors:** Temperature and humidity have a strong influence on rental counts.\n"
                )
                pdf.multi_cell(0, 10, insights)
                pdf.ln(5)

                # Business Recommendations
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Business Recommendations", ln=True)
                pdf.set_font("Arial", '', 12)
                recommendations = (
                    "- **Resource Allocation:** Increase bike availability during peak hours and seasons to meet demand.\n"
                    "- **Marketing Strategies:** Target casual users with promotions on weekends and holidays.\n"
                    "- **Weather Preparedness:** Implement dynamic pricing or incentives during adverse weather to encourage usage.\n"
                    "- **Expansion Opportunities:** Consider expanding services during high-demand periods and locations.\n"
                )
                pdf.multi_cell(0, 10, recommendations)
                pdf.ln(5)

                # Predictive Modeling Summary
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Predictive Modeling Summary", ln=True)
                pdf.set_font("Arial", '', 12)

                # Include model evaluation metrics
                model_summary = (
                    f"- **Model Used:** Pre-trained Pipeline (Scaler + Model)\n"
                    f"- **Mean Squared Error (MSE):** {mse:.2f}\n"
                    f"- **R² Score:** {r2:.2f}\n"
                    "- **Model Insights:** The model accurately predicts bike rental demand, assisting in proactive planning.\n"
                )
                pdf.multi_cell(0, 10, model_summary)
                pdf.ln(5)

                # Conclusion
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Conclusion", ln=True)
                pdf.set_font("Arial", '', 12)
                conclusion = (
                    "By leveraging data-driven insights, the bike-sharing service can enhance operational efficiency, improve user satisfaction, and increase profitability. "
                    "Continuous monitoring and analysis will enable the service to adapt to changing trends and user needs."
                )
                pdf.multi_cell(0, 10, conclusion)
                pdf.ln(5)

                # Finalize and Save the PDF
                report_path = "bike_sharing_analysis_report.pdf"
                pdf.output(report_path)

                # Provide download button
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=file,
                        file_name=report_path,
                        mime="application/pdf"
                    )
                st.success("Report generated and ready for download!")

                # Clean up the temporary PDF file
                os.remove(report_path)

            except Exception as e:
                st.error(f"Error generating report: {e}")

# ====================== Feedback Tab ======================
with tabs[6]:
    st.header('💬 Feedback')

    st.markdown("""
    **We Value Your Feedback:**
    Help us improve the Bike Sharing Analysis Dashboard by providing your valuable feedback and suggestions.
    """)

    # Feedback input
    feedback = st.text_area("Provide your feedback here:")

    # Submit feedback button
    if st.button("Submit Feedback"):
        if feedback.strip() == "":
            st.warning("Please enter your feedback before submitting.")
        else:
            # Placeholder for feedback storage (e.g., database or email)
            # Implement actual storage mechanism as needed
            st.success("Thank you for your feedback!")

# ====================== About Tab ======================
with tabs[7]:
    st.header('About')

    st.markdown("""
    ### Washington D.C. Bike Sharing Service Analysis Dashboard

    This dashboard was developed using **Streamlit**, an open-source Python library for creating interactive web applications for data science and machine learning.

    **Technologies Used:**
    - **Python**: For data manipulation and analysis.
    - **Pandas**: For data manipulation and cleaning.
    - **NumPy**: For numerical computations.
    - **Scikit-learn**: For machine learning modeling.
    - **PyCaret**: For simplifying machine learning workflows.
    - **Matplotlib & Seaborn**: For data visualization.
    - **Plotly**: For interactive visualizations.
    - **Streamlit**: For creating the web application.
    - **FPDF**: For generating PDF reports.

    **Project Objectives:**
    - To analyze the bike-sharing usage patterns in Washington D.C.
    - To understand the impact of environmental and temporal factors on bike rentals.
    - To provide recommendations to users on whether it's a good day to rent a bike.
    - To offer stakeholders actionable insights through an interactive dashboard and downloadable reports.

    **Developed By:**
    - *Your Name*
    - [LinkedIn](https://www.linkedin.com)
    - [GitHub](https://www.github.com)

    **Acknowledgments:**
    - Dataset obtained from the UCI Machine Learning Repository.
    - Inspired by various data science projects and analyses in the field.

    **Contact Us:**
    If you have any questions or feedback, please reach out at [email@example.com](mailto:email@example.com).
    """)
