# Laptop Price Prediction

## Objective
The aim of this project is to build a machine learning model to predict the price of laptops based on various features such as CPU type, GPU type, RAM size, etc. The model uses a dataset containing laptop specifications and their respective prices.

## Dataset
The dataset used for this project is a CSV file named `laptop_price_dataset.csv`. It contains several features describing different laptops and their specifications, including:

- **CPU_Type**: Type of CPU (e.g., Intel, AMD).
- **GPU_Type**: Type of GPU (e.g., Nvidia, AMD).
- **RAM**: Amount of RAM in GB.
- **Weight**: Weight of the laptop in kg.
- **ScreenSize**: Size of the laptop screen in inches.
- **Resolution**: Screen resolution (e.g., 1920x1080).
- **Price (Euro)**: The price of the laptop in Euros (target variable).

## Data Preprocessing

### Cleaning:
- Removed missing values using `.dropna()`.
- Stripped extra spaces in column names using `.str.strip()`.

### Feature Selection:
- Dropped columns that are not useful for price prediction such as:
  - Company
  - Product
  - TypeName
  - ScreenResolution
  - CPU_Company
  - GPU_Company
  - OpSys
  - Memory

### Encoding Categorical Data:
- Applied LabelEncoder to encode the `CPU_Type` and `GPU_Type` columns as numeric values since machine learning algorithms require numerical input.

### Feature and Target Split:
- Split the dataset into features (X) and target variable (y), where `y` is the Price (Euro).

## Model Development

### Linear Regression:
- Used Linear Regression as the initial model to predict laptop prices based on features.
- The model was trained using 80% of the data (training set) and tested on 20% of the data (test set).
- Evaluated using three metrics: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)**.

### Regularization:
- Applied **Ridge Regression** and **Lasso Regression** for regularization to prevent overfitting and improve model performance.
- Tested Ridge and Lasso models with different regularization strengths (alpha values), ranging from 1 to 50.
- Evaluated the models using the same metrics (**MAE**, **MSE**, **R²**) to assess performance under different alpha values.

## Evaluation

### Linear Regression:
- Metrics like **MAE**, **MSE**, and **R²** were calculated to assess the model's performance.
- A learning curve was plotted to show how the training error and validation error changed as the training set size increased.
- Residual plot and Actual vs Predicted plot were generated to visualize model performance and errors.

### Ridge Regression:
- Ridge regression was used to penalize the coefficients of less significant features, with alpha values ranging from 1 to 50.
- The best-performing model was selected based on evaluation metrics.

### Lasso Regression:
- Lasso regression was tested using various alpha values, similar to Ridge. It helps perform feature selection by shrinking less important feature coefficients to zero.

## Results

### Linear Regression:
- **MAE**, **MSE**, and **R-squared** values were displayed for the Linear Regression model.
- The model's performance was visually assessed using residuals and actual vs predicted plots.

### Ridge Regression:
- Evaluated performance for different alpha values.
- Results showed how the choice of regularization strength affected the model's predictive ability.

### Lasso Regression:
- Similar to Ridge, the performance of Lasso Regression was evaluated for different alpha values.
- This allowed comparison of how Lasso performed in comparison to Ridge regression.

## Visualizations

### Learning Curve:
- A plot of training and validation errors against training set size was displayed to understand model behavior with different amounts of data.

### Residuals Plot:
- A scatter plot of predicted values versus residuals (errors) was created to check for patterns in the prediction errors. Ideally, the residuals should be randomly scattered, indicating a good fit.

### Actual vs Predicted Plot:
- A scatter plot of actual vs predicted values was created to compare the predicted laptop prices to the true prices, helping to assess the model's accuracy visually.

## Conclusion
This project demonstrated the process of using machine learning models (Linear, Ridge, and Lasso regression) to predict laptop prices based on various specifications. By evaluating and comparing different models, the goal was to identify the most accurate model for predicting laptop prices.
