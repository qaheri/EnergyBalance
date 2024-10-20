# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Data Reading and Preparation
# Load the CSV files into pandas dataframes
country1 = pd.read_csv('country1.csv')
country2 = pd.read_csv('country2.csv')

# Calculate per capita values by dividing energy data by population
for component in ['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production',
                  'Industry', 'Transport', 'Households', 'Other', 'Energy Imports', 
                  'Energy Exports', 'Total Energy Use', 'GDP']:
    country1[f'{component} per Capita'] = country1[component] / country1['Population']
    country2[f'{component} per Capita'] = country2[component] / country2['Population']

# Display the first few rows of the updated data
print("Country 1 Data with Per Capita Values:")
print(country1.head())
print("\nCountry 2 Data with Per Capita Values:")
print(country2.head())

# Step 2: Data Visualization
sns.set(style='whitegrid')  # Set the style for the charts

# List of columns to compare per capita
components_per_capita = ['Coal Production per Capita', 'Oil Production per Capita', 'Gas Production per Capita',
                         'Renewables Production per Capita', 'Industry per Capita', 'Transport per Capita',
                         'Households per Capita', 'Other per Capita', 'Energy Imports per Capita', 
                         'Energy Exports per Capita', 'Total Energy Use per Capita']

# Plot each component year by year for both countries per capita
for component in components_per_capita:
    plt.figure(figsize=(10, 6))
    plt.plot(country1['Year'], country1[component], label='Country 1', marker='o')
    plt.plot(country2['Year'], country2[component], label='Country 2', marker='s')
    plt.title(f'Comparison of {component} between Country 1 and Country 2 (Per Capita)')
    plt.xlabel('Year')
    plt.ylabel(component)
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 3: Comparing Total Energy Use and GDP Per Capita
plt.figure(figsize=(12, 6))

# Total Energy Use per Capita comparison
plt.subplot(1, 2, 1)
plt.plot(country1['Year'], country1['Total Energy Use per Capita'], label='Country 1', marker='o')
plt.plot(country2['Year'], country2['Total Energy Use per Capita'], label='Country 2', marker='s')
plt.title('Total Energy Use per Capita Comparison')
plt.xlabel('Year')
plt.ylabel('Total Energy Use per Capita')
plt.legend()
plt.grid(True)

# GDP per Capita comparison
plt.subplot(1, 2, 2)
plt.plot(country1['Year'], country1['GDP per Capita'], label='Country 1', marker='o')
plt.plot(country2['Year'], country2['GDP per Capita'], label='Country 2', marker='s')
plt.title('GDP per Capita Comparison')
plt.xlabel('Year')
plt.ylabel('GDP per Capita')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 4: AI-Based Forecasting (Per Capita)
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

# Forecast for the next 5 years for Total Energy Use per Capita of Country 1 and Country 2
future_years, future_values_country1 = forecast_energy(country1, 'Total Energy Use per Capita')
_, future_values_country2 = forecast_energy(country2, 'Total Energy Use per Capita')

# Plot the forecasted Total Energy Use per Capita
plt.figure(figsize=(10, 6))
plt.plot(country1['Year'], country1['Total Energy Use per Capita'], label='Country 1 Historical', marker='o')
plt.plot(country2['Year'], country2['Total Energy Use per Capita'], label='Country 2 Historical', marker='s')
plt.plot(future_years, future_values_country1, label='Country 1 Forecast', linestyle='--')
plt.plot(future_years, future_values_country2, label='Country 2 Forecast', linestyle='--')
plt.title('Forecast of Total Energy Use per Capita for the Next 5 Years')
plt.xlabel('Year')
plt.ylabel('Total Energy Use per Capita')
plt.legend()
plt.grid(True)
plt.show()
