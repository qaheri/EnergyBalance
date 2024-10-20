# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

country1 = pd.read_csv('Russia.csv')
Country_1 = 'Russia'
country2 = pd.read_csv('Canada.csv')
Country_2 = 'Canada'

# Ensure the data includes natural gas, nuclear, and agriculture components
for component in ['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production', 
                  'Natural Gas Production', 'Nuclear Production',
                  'Industry', 'Transport', 'Households', 'Agriculture', 'Other', 
                  'Energy Imports', 'Energy Exports', 'Total Energy Use', 'GDP']:
    country1[f'{component} per Capita'] = country1[component] / country1['Population']
    country2[f'{component} per Capita'] = country2[component] / country2['Population']

# Forecasting function
def forecast_energy(data, feature, years_to_predict=5):
    X = data['Year'].values.reshape(-1, 1)
    y = data[feature].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 1 + years_to_predict).reshape(-1, 1)
    future_values = model.predict(future_years)

    return future_years.flatten(), future_values

sns.set(style='whitegrid')

def create_subplot_group(components, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    for idx, component in enumerate(components):
        row = idx // 2
        col = idx % 2

        # Historical data
        axes[row, col].plot(country1['Year'], country1[f'{component} per Capita'], label=f'{Country_1} Historical', marker='o')
        axes[row, col].plot(country2['Year'], country2[f'{component} per Capita'], label=f'{Country_2} Historical', marker='s')

        # Forecast data
        future_years, future_values_country1 = forecast_energy(country1, f'{component} per Capita')
        _, future_values_country2 = forecast_energy(country2, f'{component} per Capita')
        axes[row, col].plot(future_years, future_values_country1, label=f'{Country_1} Forecast', linestyle='--')
        axes[row, col].plot(future_years, future_values_country2, label=f'{Country_2} Forecast', linestyle='--')

        # Chart formatting
        axes[row, col].set_title(f'{component} per Capita')
        axes[row, col].set_xlabel('Year')
        axes[row, col].set_ylabel(f'{component} per Capita')
        axes[row, col].legend()
        axes[row, col].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
    plt.show()

def plot_3d_energy_gdp_comparison(data1, data2, country_name1, country_name2):
    # Define energy components to plot
    energy_components = ['Industry per Capita', 'Transport per Capita', 'Households per Capita', 
                         'Agriculture per Capita', 'Other per Capita']
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'x', 'D']

    # Create a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data for the first country
    for component, color, marker in zip(energy_components, colors, markers):
        ax.plot(data1['Year'], data1['GDP per Capita'], data1[component],
                label=f'{country_name1} - {component}', color=color, marker=marker, linestyle='-', linewidth=1)

    # Plot data for the second country
    for component, color, marker in zip(energy_components, colors, markers):
        ax.plot(data2['Year'], data2['GDP per Capita'], data2[component],
                label=f'{country_name2} - {component}', color=color, marker=marker, linestyle='--', linewidth=1)

    # Labeling the axes
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP per Capita')
    ax.set_zlabel('Energy Consumption per Capita')
    ax.set_title(f'3D Comparison of GDP, Year, and Energy Consumption for {country_name1} and {country_name2}')

    # Show the legend
    ax.legend()

    plt.tight_layout()
    plt.show()

# Example usage with country1 and country2 data
plot_3d_energy_gdp_comparison(country1, country2, 'Russia', 'Canada')

# Updated Group 1: Including Natural Gas and Nuclear Production
create_subplot_group(['Coal Production', 'Oil Production', 'Gas Production', 'Natural Gas Production', 
                      'Nuclear Production', 'Renewables Production'], 
                     'Energy Production per Capita Comparison')

# Group 2: Usage components per Capita including Agriculture
create_subplot_group(['Industry', 'Transport', 'Households', 'Agriculture', 'Other'], 
                     'Energy Consumption per Capita Comparison')

# Group 3: Imports, Exports, Total Energy Use, and GDP per Capita
create_subplot_group(['Energy Imports', 'Energy Exports', 'Total Energy Use', 'GDP'], 
                     'Energy Imports, Exports, and Economic Indicators per Capita Comparison')
