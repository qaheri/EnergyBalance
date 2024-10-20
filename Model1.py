# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Data Reading and Preparation
# Load the CSV files into pandas dataframes
country1 = pd.read_csv('country1_energy.csv')
country2 = pd.read_csv('country2_energy.csv')

# Display the first few rows of the data to understand the structure
print("Country 1 Data:")
print(country1.head())
print("\nCountry 2 Data:")
print(country2.head())

# Step 2: Data Visualization
sns.set(style='whitegrid')  # Set the style for the charts

# List of columns to compare
components = ['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production',
              'Industry', 'Transport', 'Households', 'Other', 'Energy Imports', 
              'Energy Exports', 'Total Energy Use']

# Plot each component year by year for both countries
for component in components:
    plt.figure(figsize=(10, 6))
    plt.plot(country1['Year'], country1[component], label='Country 1', marker='o')
    plt.plot(country2['Year'], country2[component], label='Country 2', marker='s')
    plt.title(f'Comparison of {component} between Country 1 and Country 2')
    plt.xlabel('Year')
    plt.ylabel(component)
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 3: Comparing Total Energy Use and GDP
plt.figure(figsize=(12, 6))

# Total Energy Use comparison
plt.subplot(1, 2, 1)
plt.plot(country1['Year'], country1['Total Energy Use'], label='Country 1', marker='o')
plt.plot(country2['Year'], country2['Total Energy Use'], label='Country 2', marker='s')
plt.title('Total Energy Use Comparison')
plt.xlabel('Year')
plt.ylabel('Total Energy Use')
plt.legend()
plt.grid(True)

# GDP comparison
plt.subplot(1, 2, 2)
plt.plot(country1['Year'], country1['GDP'], label='Country 1', marker='o')
plt.plot(country2['Year'], country2['GDP'], label='Country 2', marker='s')
plt.title('GDP Comparison')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 4: AI-Based Forecasting
# Function to forecast future values using linear regression
def forecast_energy(data, feature, years_to_predict=5):
    # Prepare the data for forecasting
    X = data['Year'].values.reshape(-1, 1)
    y = data[feature].values

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future values
    future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 1 + years_to_predict).reshape(-1, 1)
    future_values = model.predict(future_years)

    return future_years.flatten(), future_values

# Forecast for the next 5 years for Total Energy Use of Country 1 and Country 2
future_years, future_values_country1 = forecast_energy(country1, 'Total Energy Use')
_, future_values_country2 = forecast_energy(country2, 'Total Energy Use')

# Plot the forecasted Total Energy Use
plt.figure(figsize=(10, 6))
plt.plot(country1['Year'], country1['Total Energy Use'], label='Country 1 Historical', marker='o')
plt.plot(country2['Year'], country2['Total Energy Use'], label='Country 2 Historical', marker='s')
plt.plot(future_years, future_values_country1, label='Country 1 Forecast', linestyle='--')
plt.plot(future_years, future_values_country2, label='Country 2 Forecast', linestyle='--')
plt.title('Forecast of Total Energy Use for the Next 5 Years')
plt.xlabel('Year')
plt.ylabel('Total Energy Use')
plt.legend()
plt.grid(True)
plt.show()
