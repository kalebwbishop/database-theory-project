import pandas as pd

# Load dataset with Pandas
data = pd.read_csv("covid_data.csv")

# Define target and exclude non-feature columns
target = 'total_cases'
non_feature_columns = ['iso_code', 'continent', 'location', 'date', 'total_cases']

# Extract features by excluding non-feature columns and the target
features = data.columns.difference(non_feature_columns)

# Separate the data
X = data[features]
y = data[target]

print(data.shape)
print(X.shape)
print(y.shape)

# Remove non-numeric columns from features
X = X.select_dtypes(include=['float64', 'int64'])

# Fill missing values for numeric columns
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Save to CSV
X.to_csv("covid_features.csv", index=False, header=True)
y.to_csv("covid_target.csv", index=False, header=True)
