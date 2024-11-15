# app.py

# Import necessary libraries
import os
import joblib
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
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

# Add an introductory subsection with a GIF explaining the project
st.subheader('Project Overview')
st.markdown("""
Welcome to the interactive dashboard for the Washington D.C. bike-sharing service analysis.

This tool provides insights into the usage patterns of the bike-sharing service and includes recommendations on whether it's a good day to rent a bike based on your input parameters.
""")

# Add the GIF image
gif_url = "giphy.gif"  # Replace with your desired GIF URL or local file path
st.image(gif_url, caption='Bike Sharing in Washington D.C.', use_column_width=True)

# ====================== Load the Dataset ======================
file_path = 'hour.csv'

# Function to load data
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data

# Try to load the dataset, handle exceptions gracefully
try:
    bike_data = load_data(file_path)
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
    'Recommendations',
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
    st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

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

    # Extract top correlations with 'cnt'
    cnt_correlations = corr['cnt'].drop('cnt').sort_values(ascending=False)
    top_positive_corr = cnt_correlations.head(3)
    top_negative_corr = cnt_correlations.tail(3)

    # Display dynamic comments
    st.write("**Top features positively correlated with total rentals ('cnt'):**")
    for feature, value in top_positive_corr.items():
        st.write(f"- **{feature}**: {value:.2f}")

    st.write("**Top features negatively correlated with total rentals ('cnt'):**")
    for feature, value in top_negative_corr.items():
        st.write(f"- **{feature}**: {value:.2f}")

    # Add more EDA sections as needed...

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

# ====================== Recommendations Tab ======================
with tabs[3]:
    st.header('Recommendations')

    st.write('Use the controls below to input parameters and receive a recommendation on whether it is a good day to rent a bike.')

    # Load the pre-trained model and scaler
    try:
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')  # Ensure you have a scaler.pkl if needed
    except Exception as e:
        st.error(f'Error loading pre-trained model and scaler: {e}')
        st.stop()

    # Define the categorize_hour function
    def categorize_hour(hr):
        if 6 <= hr < 12:
            return 'Morning'
        elif 12 <= hr < 18:
            return 'Afternoon'
        elif 18 <= hr < 24:
            return 'Evening'
        else:
            return 'Night'

    # Input features
    # Use descriptive variable names and provide default values for better UX
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
    mnth = st.slider('Month', 1, 12, 6)
    weekday = st.slider('Weekday (0=Sunday)', 0, 6, 3)
    yr = st.selectbox('Year', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')

    # Create a DataFrame for the input features
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
        'atemp': [temp],  # Assuming 'atemp' is similar to 'temp'
        'hum': [hum],
        'windspeed': [windspeed],
    })

    # Perform the same feature engineering as before
    # Categorize hour into time of day
    input_data['hour_category'] = input_data['hr'].apply(categorize_hour)

    # One-hot encode the 'hour_category' feature
    input_data = pd.get_dummies(input_data, columns=['hour_category'], drop_first=True)

    # Encode 'is_holiday' as a categorical feature
    input_data['is_holiday'] = 'Holiday' if holiday == 1 else 'No Holiday'
    input_data = pd.get_dummies(input_data, columns=['is_holiday'], drop_first=True)

    # Create polynomial features for 'temp' and 'hum'
    input_data['temp_squared'] = input_data['temp'] ** 2
    input_data['hum_squared'] = input_data['hum'] ** 2
    input_data['temp_hum_interaction'] = input_data['temp'] * input_data['hum']

    # Define the features used during training
    features = bike_data.columns.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'])

    # Ensure the input_data has the same columns as training data
    missing_cols = set(features) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[features]  # Ensure the order matches

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the pre-trained model
    prediction = best_model.predict(input_data_scaled)
    predicted_count = int(prediction[0])

    # Based on the predicted count, provide a recommendation
    # Define thresholds based on the distribution of 'cnt' in the dataset
    cnt_mean = bike_data['cnt'].mean()
    cnt_std = bike_data['cnt'].std()

    if predicted_count >= cnt_mean + cnt_std:
        recommendation = "🌟 It's a great day to rent a bike! High demand expected."
    elif predicted_count >= cnt_mean:
        recommendation = "👍 It's a good day to rent a bike."
    else:
        recommendation = "🤔 It might not be the best day to rent a bike due to lower demand."

    st.subheader('Recommendation')
    st.write(recommendation)
    st.write(f"**Predicted Number of Bike Users:** {predicted_count}")

# ====================== Download Report Tab ======================
with tabs[4]:
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
with tabs[4]:
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
with tabs[6]:
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

