# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

country1 = pd.read_csv('country1.csv')
Country_1 = 'Russia'
country2 = pd.read_csv('country2.csv')
Country_2 = 'Canada'

for component in ['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production',
                  'Industry', 'Transport', 'Households', 'Other', 'Energy Imports', 
                  'Energy Exports', 'Total Energy Use', 'GDP']:
    country1[f'{component} per Capita'] = country1[component] / country1['Population']
    country2[f'{component} per Capita'] = country2[component] / country2['Population']

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

# Group 1: Production components per Capita
create_subplot_group(['Coal Production', 'Oil Production', 'Gas Production', 'Renewables Production'], 
                     'Energy Production per Capita Comparison 1')

# Group 2: Usage components per Capita
create_subplot_group(['Industry', 'Transport', 'Households', 'Other'], 
                     'Energy Consumption per Capita Comparison 2')

# Group 3: Imports, Exports, Total Energy Use, and GDP per Capita
create_subplot_group(['Energy Imports', 'Energy Exports', 'Total Energy Use', 'GDP'], 
                     'Energy Imports, Exports, and Economic Indicators per Capita Comparison')
