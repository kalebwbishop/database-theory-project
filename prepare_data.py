import pandas as pd

# Load dataset with Pandas
data = pd.read_csv("covid_data.csv")

# Extract features and target
features = ['population_density', 'gdp_per_capita', 'hospital_beds_per_thousand', 'new_tests', 'stringency_index']
target = 'new_cases'

# Separate the data
X = data[features]
y = data[target]

X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Save to CSV
X.to_csv("covid_data_features.csv", index=False, header=True)
y.to_csv("covid_data_target.csv", index=False, header=True)