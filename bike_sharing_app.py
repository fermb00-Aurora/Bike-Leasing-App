import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the Dataset
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

# Streamlit Configuration
st.set_page_config(page_title='Bike Leasing Company Dashboard', page_icon='🚲', layout='wide', initial_sidebar_state='expanded')
st.title('Bike Leasing Company - Data Analysis & Insights')

# Tabs for ML Models
tabs = st.tabs(['Exploratory Analysis', 'Linear Models', 'Non-Linear Models', 'Model Comparison'])

# Exploratory Analysis Tab
with tabs[0]:
    st.header('Exploratory Data Analysis')

    # Checking for Missing Values
    st.subheader('Missing Values')
    missing_values = bike_data.isnull().sum()
    st.write(missing_values[missing_values > 0])
    st.caption("Checking for missing values ensures data quality and completeness.")

    # Data Quality Check - Duplicates
    st.subheader('Duplicate Rows')
    duplicates = bike_data.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

    # Outliers Detection
    st.subheader('Outliers Detection')
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    fig, axes = plt.subplots(1, len(numerical_features), figsize=(15, 5))
    for idx, feature in enumerate(numerical_features):
        sns.boxplot(data=bike_data, y=feature, ax=axes[idx], color='skyblue')
        axes[idx].set_title(feature)
    st.pyplot(fig)
    st.caption("Boxplots help detect outliers in numerical features.")

    # Feature Correlation
    st.subheader('Feature Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(bike_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.caption("Correlation heatmap helps identify which features are most related to bike rentals.")

    # Feature Engineering
    st.subheader('Feature Engineering')
    bike_data['hour_category'] = bike_data['hr'].apply(lambda x: 'Morning' if 6 <= x < 12 else 'Afternoon' if 12 <= x < 18 else 'Evening' if 18 <= x < 24 else 'Night')
    st.write("New feature 'hour_category' created to capture different times of the day.")
    st.write(bike_data[['hr', 'hour_category']].head())

    # Year-wise Bike Rentals
    st.subheader('Year-wise Bike Rentals')
    fig = px.bar(bike_data, x='yr', y='cnt', color='yr', title='Bike Rentals by Year', labels={'yr': 'Year', 'cnt': 'Bike Rentals'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This graph helps understand the overall growth or decline in bike rentals year-over-year.")

# Linear Models Tab
with tabs[1]:
    st.header("Linear Regression Models")

    # Selecting Features and Target
    features = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
        'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
    ]
    target = 'cnt'

    # Split Data
    X = bike_data[features]
    y = bike_data[target]
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Train Ridge Regression Model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # Train Lasso Regression Model
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Display Evaluation Metrics
    st.subheader("Model Evaluation - Linear Models")
    st.write(f"Linear Regression - Mean Squared Error: {mse_lr:.2f}, R² Score: {r2_lr:.2f}")
    st.write(f"Ridge Regression - Mean Squared Error: {mse_ridge:.2f}, R² Score: {r2_ridge:.2f}")
    st.write(f"Lasso Regression - Mean Squared Error: {mse_lasso:.2f}, R² Score: {r2_lasso:.2f}")

# Non-Linear Models Tab
with tabs[2]:
    st.header("Non-Linear Models")

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Train Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    # Display Evaluation Metrics
    st.subheader("Model Evaluation - Non-Linear Models")
    st.write(f"Random Forest - Mean Squared Error: {mse_rf:.2f}, R² Score: {r2_rf:.2f}")
    st.write(f"Gradient Boosting - Mean Squared Error: {mse_gb:.2f}, R² Score: {r2_gb:.2f}")

# Model Comparison Tab
with tabs[3]:
    st.header("Model Comparison")

    st.subheader("Linear vs Non-Linear Models")
    models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'Gradient Boosting']
    r2_scores = [r2_lr, r2_ridge, r2_lasso, r2_rf, r2_gb]
    mse_scores = [mse_lr, mse_ridge, mse_lasso, mse_rf, mse_gb]

    comparison_data = pd.DataFrame({
        'Model': models,
        'R² Score': r2_scores,
        'Mean Squared Error': mse_scores
    })

    # R² Score Comparison
    fig_r2 = px.bar(comparison_data, x='Model', y='R² Score', title='Model Comparison - R² Score', color='Model', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_r2, use_container_width=True)

    # Mean Squared Error Comparison
    fig_mse = px.bar(comparison_data, x='Model', y='Mean Squared Error', title='Model Comparison - Mean Squared Error', color='Model', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_mse, use_container_width=True)

    # Conclusion: Best Model Selection
    best_model_index = np.argmax(r2_scores)
    st.write(f"The best model based on R² Score is: **{models[best_model_index]}** with an R² Score of **{r2_scores[best_model_index]:.2f}**.")

# To run this Streamlit app, use the command: streamlit run <script_name.py>

