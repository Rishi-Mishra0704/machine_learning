# USA Housing Price Prediction

## Objective
The goal of this project is to build a machine learning model to predict the housing prices based on various features such as area income, house age, number of rooms, number of bedrooms, and area population. The model uses a housing dataset to predict the prices of homes.

## Dataset
The dataset used for this project is a CSV file named `USA_Housing.csv`. It contains several features describing different houses, including:

- **Avg. Area Income**: Average income in the area.
- **Avg. Area House Age**: Average age of houses in the area.
- **Avg. Area Number of Rooms**: Average number of rooms in houses in the area.
- **Avg. Area Number of Bedrooms**: Average number of bedrooms in houses in the area.
- **Area Population**: Population in the area.
- **Price**: The price of the house (target variable).

## Data Preprocessing

### Cleaning:
- Dropped the `Address` column as it contained string values not useful for prediction.
- Performed basic data checks using `.info()` and `.describe()` to identify data distribution and any missing values.

### Feature Selection:
- Kept relevant features like area income, house age, number of rooms, number of bedrooms, and population for predicting housing prices.
- Dropped the `Address` column since it didn't contribute to numerical prediction.

### Data Visualization:
- Plotted a pairplot to visualize the relationships between features.
- Plotted the distribution of housing prices using `sns.displot()` for better understanding of the target variable.
- Created a heatmap of correlations between features to visualize how strongly they are related to each other.

### Splitting Data:
- Split the dataset into features (X) and target (y) for model training.
  - Features (`X`): Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population.
  - Target (`y`): Price.

## Model Development

### Linear Regression:
- Used **Linear Regression** to predict housing prices based on the selected features.
- Split the dataset into training (60%) and testing (40%) sets.
- Trained the model on the training data and evaluated its performance on the test data.

### Model Evaluation:
- Evaluated the model using metrics like:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared (R²)**

### Interpretation:
- **Intercept and Coefficients**: The intercept represents the baseline price when all features are zero. The coefficients provide insight into how each feature influences the price.
- Evaluated the relationship between predictions and actual prices using scatter plots and residuals distribution.

## Results

### Model Performance:
- The model performed reasonably well with the given features, but more tuning and additional features might improve accuracy.
- The **R-squared value** indicated the proportion of variance explained by the model.

### Metrics:
- **MAE**: Mean Absolute Error between predicted and actual values.
- **MSE**: Mean Squared Error to measure the average of the squares of errors.
- **RMSE**: The square root of the MSE to get the error in the original units.
- **R²**: The coefficient of determination, indicating how well the model explains the variance in the data.

## Visualizations

### Correlation Heatmap:
- The heatmap shows the correlation between different features. Features like `Avg. Area Income` and `Avg. Area Number of Rooms` are highly correlated with `Price`.

### Predictions vs Actual Plot:
- A scatter plot of predicted vs actual housing prices was created to visually assess model accuracy.

### Residuals Distribution:
- A distribution plot of residuals (errors between predicted and actual prices) was generated to check for any patterns in prediction errors. Ideally, the residuals should be randomly distributed.

## Conclusion
This project demonstrates how to use machine learning, specifically **Linear Regression**, to predict housing prices based on various area features. The model shows reasonable accuracy, but there is potential for further improvement with additional features and model tuning.
