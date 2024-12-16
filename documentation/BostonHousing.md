# Boston House Price Prediction

## Objective
The aim of this project is to build a machine learning model to predict the price of houses in Boston based on various features such as crime rate, average number of rooms, and more. The model uses a dataset containing house features and their respective prices.

## Dataset
The dataset used for this project is a CSV file named `BostonHousing.csv`. It contains several features describing different houses and their characteristics, including:

- **crim**: Crime rate (per capita).
- **zn**: Proportion of residential land zoned for large-scale buildings.
- **indus**: Proportion of non-retail business acres per town.
- **lstat**: Percentage of the population with low socio-economic status.
- **ptratio**: Pupil-teacher ratio by town.
- **rm**: Average number of rooms per dwelling.
- **medv**: Median value of owner-occupied homes in $1000s (target variable).

## Data Preprocessing

### Cleaning:
- Removed missing values using `.dropna()`.
- Stripped extra spaces in column names using `.str.strip()`.
- Dropped duplicates from the dataset.

### Feature Selection:
- Selected relevant features for prediction, such as:
  - `rm`, `zn`, `indus`, `lstat`, `ptratio`, and `crim`.

### Feature and Target Split:
- Split the dataset into features (X) and target variable (y), where y is the `medv` (median value of homes).

## Model Development

### Linear Regression:
- Used Linear Regression as the initial model to predict house prices based on features.
- The model was trained using 80% of the data (training set) and tested on 20% of the data (test set).
- Evaluated using two metrics: **Mean Squared Error (MSE)** and **R-squared (R²)**.

### Polynomial Regression:
- Applied polynomial transformation to the features to capture non-linear relationships between features and the target variable.
- Polynomial features of degree 2 were used.

### Regularization:
- Applied **Ridge Regression** and **Lasso Regression** for regularization to prevent overfitting and improve model performance.
- Tested Ridge and Lasso models with different regularization strengths (alpha values).
- Evaluated models using **MSE** and **R²**.

## Evaluation

### Linear Regression:
- Metrics like **MSE** and **R²** were calculated to assess the model's performance.
- A learning curve was plotted to show how training and validation errors changed with varying training set sizes.
- Residual plot and Actual vs Predicted plot were generated to visualize model performance and errors.

### Ridge Regression:
- Ridge regression was used to penalize coefficients of less significant features.
- The best-performing Ridge model was selected based on evaluation metrics.

### Lasso Regression:
- Lasso regression was tested to shrink less important feature coefficients to zero, performing implicit feature selection.

## Results

### Linear Regression:
- **MSE**: 10.75
- **R²**: 0.85
- Cross-validation mean score: 0.46

### Ridge Regression:
- **MSE**: 10.41
- **R²**: 0.86
- Cross-validation mean score: 0.53

### Lasso Regression:
- **MSE**: 17.01
- **R²**: 0.77
- Cross-validation mean score: 0.48

## Visualizations

### Learning Curve:
- A plot of training and validation errors against the training set size was displayed to understand model behavior with different amounts of data.

### Residuals Plot:
- A scatter plot of predicted values versus residuals (errors) was created to check for patterns in prediction errors. Ideally, residuals should be randomly scattered, indicating a good fit.

### Actual vs Predicted Plot:
- A scatter plot of actual vs predicted values was created to compare the predicted house prices to the true values, helping to assess the model's accuracy visually.

## Conclusion
This project demonstrated the process of using machine learning models (Linear, Ridge, and Lasso regression) to predict house prices based on various features. By evaluating and comparing different models, the goal was to identify the most accurate model for predicting house prices.
