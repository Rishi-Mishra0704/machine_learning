import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('../data/laptop_price_dataset.csv')
df.head()

print(df.isnull().sum())
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Price (Euro)', axis=1)
y = df['Price (Euro)']

# Split the data into training (60%) and temporary set (40%)
train, temp = train_test_split(df, test_size=0.4, random_state=42,)
# Split the temporary set into validation (50% of temp = 20%) and test (50% of temp = 20%)
val, test = train_test_split(temp, test_size=0.5, random_state=42,)

# Verify the split, uncomment the following lines
# print(f"Train set size: {train.shape}")
# print(f"Validation set size: {val.shape}")
# print(f"Test set size: {test.shape}")

# Standardize the data, Separating the features from the target
X_train = train.drop('Price (Euro)', axis=1)
y_train = train['Price (Euro)']
X_val = val.drop('Price (Euro)', axis=1)
y_val = val['Price (Euro)']
X_test = test.drop('Price (Euro)', axis=1)
y_test = test['Price (Euro)']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(X_train_scaled, y_train)
#check the model score
# print(f"Model coefficients: {LinearRegressionModel.coef_}")
# print(f"Intercept: {LinearRegressionModel.intercept_}")

# Make predictions
y_pred_train = LinearRegressionModel.predict(X_train_scaled)
y_pred_val = LinearRegressionModel.predict(X_val_scaled)
y_pred_test = LinearRegressionModel.predict(X_test_scaled)

# Print a few predictions to check
print(f"Predictions on the training set: {y_pred_train[:5]}")

# Evaluate on the training set
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
# Evaluate on the validation set
mae_val = mean_absolute_error(y_val, y_pred_val)
mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
# Evaluate on the test set
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train MAE: {mae_train}, MSE: {mse_train}, R²: {r2_train}")
print(f"Validation MAE: {mae_val}, MSE: {mse_val}, R²: {r2_val}")
print(f"Test MAE: {mae_test}, MSE: {mse_test}, R²: {r2_test}")

