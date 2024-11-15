# Import necessary libraries
import os
import joblib
import warnings

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
from sklearn.preprocessing import StandardScaler
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
    'Feedback'
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
    st.write(missing_values[missing_values > 0])

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

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')

    # Calculate the correlation matrix
    corr = bike_data.corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm')
    st.pyplot(fig)

    # Insights on features relevant for prediction
    st.write('Features like "temp", "hum", "hr", "season", and "weathersit" show strong correlation with "cnt".')

    # Bike Counts over Hours
    st.subheader('Bike Counts over Hours')

    # Calculate average bike counts per hour
    hour_counts = bike_data.groupby('hr')['cnt'].mean()

    # Plot the average bike counts per hour
    fig = px.line(
        x=hour_counts.index,
        y=hour_counts.values,
        labels={'x': 'Hour of the Day', 'y': 'Average Count'},
        title='Average Bike Count per Hour'
    )
    st.plotly_chart(fig)

    # Interactive Feature Exploration
    st.subheader('Interactive Feature Exploration')

    # Allow users to select features for x and y axes
    x_axis = st.selectbox('Select X-axis', options=bike_data.columns)
    y_axis = st.selectbox('Select Y-axis', options=bike_data.columns)

    # Plot the selected features
    fig = px.scatter(bike_data, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
    st.plotly_chart(fig)

    # Additional Insights
    st.write('From the plots, we observe peak usage during rush hours, indicating that many users are commuting to work.')

# ====================== Predictive Modeling Tab ======================
with tabs[3]:
    st.header('Predictive Modeling')

    # Load the pre-trained model and scaler
    st.subheader('Load Pre-trained Model')

    try:
        # Load the scaler
        scaler = joblib.load('scaler.pkl')
        # Load the pre-trained best model
        best_model = joblib.load('best_model.pkl')
        st.success('Pre-trained model and scaler loaded successfully.')
    except Exception as e:
        st.error(f'Error loading pre-trained model and scaler: {e}')
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

    # Feature Scaling
    st.write('Scaling features using the loaded StandardScaler.')

    # Transform the features using the loaded scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Evaluation
    st.subheader('Model Evaluation')

    # Use the pre-trained model to make predictions on the test set
    y_pred = best_model.predict(X_test_scaled)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Loaded Model MSE: {mse:.2f}, R²: {r2:.2f}')

    # Plotting predictions vs actual values
    st.subheader('Predictions vs Actual Values')

    # Plot actual vs predicted values
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title='Actual vs Predicted Bike Counts'
    )
    st.plotly_chart(fig)

# ====================== Simulator Tab ======================
with tabs[4]:
    st.header('Bike Usage Prediction Simulator')

    st.write('Use the controls below to input parameters and predict the expected number of bike users.')

    # Load the pre-trained model and scaler
    try:
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
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
        format_func=lambda x: {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain'}[x]
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
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the pre-trained model
    prediction = best_model.predict(input_data_scaled)

    st.subheader(f'Predicted Number of Bike Users: **{int(prediction[0])}**')

# ====================== Download Report Tab ======================
with tabs[5]:
    st.header('📄 Download Report')

    st.markdown("""
    **Generate and Download a Professional PDF Report:**
    Compile your analysis and model evaluation results into a concise and professional PDF report for offline review and sharing with stakeholders.
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
                    "This report provides a comprehensive analysis of the Washington D.C. bike-sharing service. "
                    "It includes data overview, exploratory data analysis, predictive modeling, and insights into usage patterns."
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
                    "- **Features:** The dataset includes hourly rental data and weather information.\n"
                    "- **Time Period:** Covers bike-sharing data over two years.\n"
                )
                pdf.multi_cell(0, 10, data_overview)
                pdf.ln(5)

                # Exploratory Data Analysis
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Exploratory Data Analysis", ln=True)
                pdf.set_font("Arial", '', 12)
                eda_summary = (
                    "- **Peak Usage Hours:** Bike rentals peak during morning and evening rush hours.\n"
                    "- **Seasonal Trends:** Higher usage observed during warmer months.\n"
                    "- **Weather Impact:** Adverse weather conditions lead to decreased bike usage.\n"
                )
                pdf.multi_cell(0, 10, eda_summary)
                pdf.ln(5)

                # Predictive Modeling Summary
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Predictive Modeling Summary", ln=True)
                pdf.set_font("Arial", '', 12)

                # Include model evaluation metrics
                model_summary = (
                    f"- **Model Used:** Pre-trained Model ({type(best_model).__name__})\n"
                    f"- **Mean Squared Error (MSE):** {mse:.2f}\n"
                    f"- **R² Score:** {r2:.2f}\n"
                    "- **Model Insights:** The pre-trained model provides reliable predictions of bike usage.\n"
                )
                pdf.multi_cell(0, 10, model_summary)
                pdf.ln(5)

                # Conclusion
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Conclusion", ln=True)
                pdf.set_font("Arial", '', 12)
                conclusion = (
                    "The analysis reveals significant patterns in bike-sharing usage related to time of day, season, and weather conditions. "
                    "The predictive model can assist in forecasting demand and optimizing resource allocation."
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
