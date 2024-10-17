import pandas as pd

country1_data = {
    'Year': [2020, 2021, 2022, 2023],
    'Coal Production': [150, 160, 155, 165],
    'Oil Production': [120, 130, 125, 140],
    'Gas Production': [80, 85, 90, 95],
    'Renewables Production': [60, 70, 75, 80],
    'Industry': [100, 110, 115, 120],
    'Transport': [50, 55, 60, 65],
    'Households': [40, 45, 50, 55],
    'Other': [30, 35, 40, 45],
    'Energy Imports': [200, 210, 220, 230],
    'Energy Exports': [50, 55, 60, 65],
    'Total Energy Use': [420, 440, 460, 480],
    'GDP': [30000, 32000, 34000, 36000]
}

country2_data = {
    'Year': [2020, 2021, 2022, 2023],
    'Coal Production': [100, 110, 105, 115],
    'Oil Production': [90, 95, 100, 105],
    'Gas Production': [70, 75, 80, 85],
    'Renewables Production': [50, 60, 65, 70],
    'Industry': [80, 85, 90, 95],
    'Transport': [40, 45, 50, 55],
    'Households': [30, 35, 40, 45],
    'Other': [20, 25, 30, 35],
    'Energy Imports': [150, 160, 170, 180],
    'Energy Exports': [40, 45, 50, 55],
    'Total Energy Use': [360, 380, 400, 420],
    'GDP': [25000, 26000, 27000, 28000]
}

country1_df = pd.DataFrame(country1_data)
country2_df = pd.DataFrame(country2_data)

country1_file_path = '/mnt/data/country1_energy.csv'
country2_file_path = '/mnt/data/country2_energy.csv'

country1_df.to_csv(country1_file_path, index=False)
country2_df.to_csv(country2_file_path, index=False)

country1_file_path, country2_file_path
