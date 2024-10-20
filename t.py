import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

country1_data = pd.read_csv('country1_energy.csv')
country2_data = pd.read_csv('country2_energy.csv')

country1_data.fillna(0, inplace=True)
country2_data.fillna(0, inplace=True)

print("Country 1 Columns:", country1_data.columns)
print("Country 2 Columns:", country2_data.columns)

#Energy Production Comparison (Fossil Fuels, Renewables)
labels = ['Coal', 'Oil', 'Gas', 'Renewables']

country1_prod = [
    country1_data['Coal Production'].sum(),
    country1_data['Oil Production'].sum(),
    country1_data['Gas Production'].sum(),
    country1_data['Renewables Production'].sum()
]

country2_prod = [
    country2_data['Coal Production'].sum(),
    country2_data['Oil Production'].sum(),
    country2_data['Gas Production'].sum(),
    country2_data['Renewables Production'].sum()
]

# Plot energy production comparison
x = range(len(labels))
plt.figure(figsize=(10, 6))
plt.bar(x, country1_prod, width=0.4, label='Country 1', align='center')
plt.bar([p + 0.4 for p in x], country2_prod, width=0.4, label='Country 2', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.ylabel('Energy Production (TWh)')
plt.title('Energy Production Comparison by Source')
plt.legend()
plt.show()

# Energy Consumption by Sector
labels = ['Industry', 'Transport', 'Households', 'Other']

country1_consumption = [
    country1_data['Industry'].sum(),
    country1_data['Transport'].sum(),
    country1_data['Households'].sum(),
    country1_data['Other'].sum()
]

country2_consumption = [
    country2_data['Industry'].sum(),
    country2_data['Transport'].sum(),
    country2_data['Households'].sum(),
    country2_data['Other'].sum()
]

# Plot energy consumption comparison
plt.figure(figsize=(10, 6))
plt.bar(x, country1_consumption, width=0.4, label='Country 1', align='center')
plt.bar([p + 0.4 for p in x], country2_consumption, width=0.4, label='Country 2', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.ylabel('Energy Consumption (TWh)')
plt.title('Energy Consumption by Sector')
plt.legend()
plt.show()

# Energy Imports/Exports Comparison

labels = ['Imports', 'Exports']

country1_trade = [
    country1_data['Energy Imports'].sum(),
    country1_data['Energy Exports'].sum()
]

country2_trade = [
    country2_data['Energy Imports'].sum(),
    country2_data['Energy Exports'].sum()
]

x_trade = range(len(labels))

# Plot energy imports/exports comparison
plt.figure(figsize=(10, 6))
plt.bar(x_trade, country1_trade, width=0.4, label='Country 1', align='center')
plt.bar([p + 0.4 for p in x_trade], country2_trade, width=0.4, label='Country 2', align='center')
plt.xticks([p + 0.2 for p in x_trade], labels)
plt.ylabel('Energy Trade (TWh)')
plt.title('Energy Imports/Exports Comparison')
plt.legend()
plt.show()

# 4. Energy Efficiency (Energy Use per GDP Unit)

country1_efficiency = country1_data['Total Energy Use'].sum() / country1_data['GDP'].sum()
country2_efficiency = country2_data['Total Energy Use'].sum() / country2_data['GDP'].sum()

# Plot energy efficiency comparison
plt.figure(figsize=(6, 4))
plt.bar(['Country 1', 'Country 2'], [country1_efficiency, country2_efficiency])
plt.ylabel('Energy Use per GDP Unit (TWh/USD)')
plt.title('Energy Efficiency Comparison')
plt.show()

# 5. AI: Forecasting Future Energy Consumption (Linear Regression)

# Forecasting energy consumption using Linear Regression

# Country 1 Forecast
X1 = country1_data[['Year']]
y1 = country1_data['Total Energy Consumption']  # Check this column name for spelling and case sensitivity

# Country 2 Forecast
X2 = country2_data[['Year']]
y2 = country2_data['Total Energy Consumption']  # Check this column name for spelling and case sensitivity

# Train-Test Split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)

# Train Linear Regression Model
model1 = LinearRegression().fit(X1_train, y1_train)
model2 = LinearRegression().fit(X2_train, y2_train)

# Predict future years (e.g., 2025, 2030, 2035)
future_years = [[2025], [2030], [2035]]
country1_pred = model1.predict(future_years)
country2_pred = model2.predict(future_years)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot([2025, 2030, 2035], country1_pred, label='Country 1 Prediction', marker='o')
plt.plot([2025, 2030, 2035], country2_pred, label='Country 2 Prediction', marker='o')
plt.xlabel('Year')
plt.ylabel('Energy Consumption (TWh)')
plt.title('Predicted Energy Consumption')
plt.legend()
plt.show()
