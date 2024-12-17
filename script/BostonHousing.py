import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'BostonHousing.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

#cleaning data
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()


# Visualizing the distributions of numerical features
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_cleaned.columns):
    plt.subplot(3, 5, i+1)
    sns.histplot(df_cleaned[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Visualizing the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# Define features (X) and target (y)
X = df[['rm', 'zn', "indus", 'lstat', 'ptratio', "crim",]].dropna()
y = df.loc[X.index,'medv'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_poly, y_train)

y_pred = linear_model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Evaluation")
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Line for perfect prediction
plt.xlabel('Actual values (medv)')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted Values')
plt.show()

model_pipeline = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Perform cross-validation using the pipeline
cv_scores = cross_val_score(model_pipeline, X, y, cv=5)

# Print cross-validation results
print("Linear Regression")
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

# Regularization with Ridge Regression
ridge_pipeline = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10))

# Perform cross-validation using the pipeline
cv_scores = cross_val_score(ridge_pipeline, X, y, cv=5)

# Print cross-validation results
print("Ridge Regression")
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

ridge_pred = Ridge(alpha=10).fit(X_train_poly, y_train).predict(X_test_poly)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Regression Evaluation")
print(f'Mean Squared Error: {ridge_mse:.2f}')
print(f'R^2 Score: {ridge_r2:.2f}')


# Regularization with Lasso Regression
lasso_pipeline = make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=10))

# Perform cross-validation using the pipeline
cv_scores = cross_val_score(lasso_pipeline, X, y, cv=5)
print("Lasso Regression")
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

lasso_pred = Lasso(alpha=10).fit(X_train_poly, y_train).predict(X_test_poly)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("Lasso Regression Evaluation")
print(f'Mean Squared Error: {lasso_mse:.2f}')
print(f'R^2 Score: {lasso_r2:.2f}')

