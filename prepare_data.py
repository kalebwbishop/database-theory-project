import pandas as pd

# Load dataset with Pandas
data = pd.read_csv("laptop_data.csv")

# Define target and exclude non-feature columns
target = 'Price (Euro)'
non_feature_columns = [target]

# Separate features and target
features = data.columns.difference(non_feature_columns)
X = data[features]
y = data[target]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding with integer dtype (1s and 0s)
X_encoded = pd.get_dummies(X, columns=categorical_cols, dtype=int)

# Save to CSV
X_encoded.to_csv("laptop_features.csv", index=False, header=True)
y.to_csv("laptop_target.csv", index=False, header=True)
