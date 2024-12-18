import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import learning_curve

# +
working_dir = os.getcwd()

csv_path = os.path.join(working_dir,"..","data", 'laptop_price_dataset.csv')
df = pd.read_csv(csv_path)
df.head()

# -

df = df.dropna()

# clean the data
df.columns = df.columns.str.strip()

columns_to_drop = ['Company', 'Product', 'TypeName', 'ScreenResolution', 
                   'CPU_Company', 'GPU_Company', 'OpSys', 'Memory']

# Drop the specified columns
df_cleaned = df.drop(columns=columns_to_drop)

# Show the resulting DataFrame
df_cleaned.head()

X = df_cleaned[[
    "Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)",
]]
y = df_cleaned['Price (Euro)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Residual Plot

y_pred = linear_regression.predict(X_test)

# Plotting residuals
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Line at 0 residuals
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Ideal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Define a range of alpha values to test
alpha_values = [1, 10, 20, 50]

# Initialize lists to store evaluation metrics for each alpha
results = []

# Loop through alpha values and evaluate the Ridge model
for alpha in alpha_values:
    # Create Ridge model with the current alpha
    ridge_model = Ridge(alpha=alpha)
    
    # Fit the model on the training data
    ridge_model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred_ridge = ridge_model.predict(X_test)
    
    # Evaluate the model
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    # Append the results
    results.append({
        "Alpha": alpha,
        "MAE": mae_ridge,
        "MSE": mse_ridge,
        "R2": r2_ridge
    })

# Print results for each alpha
for res in results:
    print(f"Alpha: {res['Alpha']}, MAE: {res['MAE']:.2f}, MSE: {res['MSE']:.2f}, R-squared: {res['R2']:.2f} in Ridge Regression")


# Regularization using Lasso
alpha_values = [1, 10, 20, 50]

# Initialize lists to store evaluation metrics for each alpha
results = []
# Create the Lasso model with a regularization parameter alpha (lambda)
for alpha in alpha_values:
    # Create Ridge model with the current alpha
    lasso_model = Lasso(alpha=alpha)
    
    # Fit the model on the training data
    lasso_model.fit(X_train, y_train)

    # Predict on the test data
    y_pred_lasso = lasso_model.predict(X_test)

    # Evaluate the model
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
        
        # Append the results
    results.append({
            "Alpha": alpha,
            "MAE": mae_lasso,
            "MSE": mse_lasso,
            "R2": r2_lasso
        })

# Print results for each alpha
for res in results:
    print(f"Alpha: {res['Alpha']}, MAE: {res['MAE']:.2f}, MSE: {res['MSE']:.2f}, R-squared: {res['R2']:.2f} for Lasso")

# +
test_data = pd.DataFrame({
    "Inches": [15.6, 14.0, 17.3, 13.3],
    "CPU_Frequency (GHz)": [2.5, 3.1, 2.9, 3.8],
    "RAM (GB)": [8, 16, 32, 4],
    "Weight (kg)": [2.1, 1.5, 3.2, 1.2],
})


# Perform predictions
predicted_prices = linear_regression.predict(test_data)

# Show results
print("Test Data:")
print(test_data)
print("\nPredicted Prices:")
print(predicted_prices)
