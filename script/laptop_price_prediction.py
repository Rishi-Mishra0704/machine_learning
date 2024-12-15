import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import learning_curve

df = pd.read_csv('../data/laptop_price_dataset.csv')
df.head()

df = df.dropna()

# clean the data
df.columns = df.columns.str.strip()

columns_to_drop = ['Company', 'Product', 'TypeName', 'ScreenResolution', 
                   'CPU_Company', 'GPU_Company', 'OpSys', 'Memory']

# Drop the specified columns
df_cleaned = df.drop(columns=columns_to_drop)

# Show the resulting DataFrame
df_cleaned.head()

X = df_cleaned.drop('Price (Euro)', axis=1)  # Features
y = df_cleaned['Price (Euro)'] 

label_encoder = LabelEncoder()

X["CPU_Type"] = label_encoder.fit_transform(X["CPU_Type"])
X["GPU_Type"] = label_encoder.fit_transform(X["GPU_Type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predict the test set
y_pred = linear_regression.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae} in Linear Regression")
print(f"Mean Squared Error: {mse} in Linear Regression")
print(f"R-squared: {r2} in Linear Regression")

# Learning Curve
train_sizes, train_scores, validation_scores = learning_curve(linear_regression, X_train, y_train, cv=5)

# Plotting
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training error')
plt.plot(train_sizes, validation_scores.mean(axis=1), label='Validation error')
plt.xlabel('Training size')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.legend()
plt.show()

#Residual Plot

y_pred = linear_regression.predict(X_test)

# Plotting residuals
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Line at 0 residuals
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

print(residuals)

# Actual vs Predicted

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Ideal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
print("y_test" , y_test)
print("y_pred" , y_pred)

# Regularization using Ridge

# Create the Ridge model with a regularization parameter alpha (lambda)
ridge_model = Ridge(alpha=1.0)  # alpha is the regularization parameter (lambda)

# Fit the model on the training data
ridge_model.fit(X_train, y_train)

# Predict on the test data
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - MAE: {mae_ridge:.2f}, MSE: {mse_ridge:.2f}, R-squared: {r2_ridge:.2f}")


# Regularization using Lasso

# Create the Lasso model with a regularization parameter alpha (lambda)
lasso_model = Lasso(alpha=0.1)  # alpha is the regularization parameter (lambda)

# Fit the model on the training data
lasso_model.fit(X_train, y_train)

# Predict on the test data
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the model
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso Regression - MAE: {mae_lasso:.2f}, MSE: {mse_lasso:.2f}, R-squared: {r2_lasso:.2f}")



