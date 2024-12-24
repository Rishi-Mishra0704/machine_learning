# House Price Prediction using Neural Networks

## Objective
The objective of this project is to predict house prices based on various features such as square footage, number of bedrooms, location, and more using a neural network model. The dataset is related to house sales, and the model will use features to predict the price of houses.

## Dataset
The dataset used is `kc_house_data.csv` and contains various columns such as:
- **id**: Unique identifier for each house.
- **date**: Date when the house was sold.
- **price**: Price of the house (target variable).
- **sqft_living**: Square footage of the living space.
- **sqft_lot**: Square footage of the lot.
- **bedrooms**: Number of bedrooms.
- **bathrooms**: Number of bathrooms.
- **sqft_above**: Square footage of the house excluding the basement.
- **sqft_basement**: Square footage of the basement.
- **waterfront**: Whether the house is on the waterfront.
- **view**: View quality rating.
- **condition**: Condition of the house.
- **grade**: Grade of the house.
- **sqft_living15**: Square footage of the house in the past 15 years.
- **sqft_lot15**: Square footage of the lot in the past 15 years.
- **lat**: Latitude of the house.
- **long**: Longitude of the house.
- **zipcode**: Postal code of the house.

## Data Preprocessing

### Cleaning:
- Checked for missing values in the dataset, and confirmed there were no significant issues.
- Removed the `id` and `zipcode` columns as they were not needed for prediction.
- Converted the `date` column to datetime and extracted the year and month into separate columns.

### Feature Engineering:
- Created new features such as `year` and `month` from the `date` column.
- Dropped the `date` and `zipcode` columns after extracting the required features.
  
### Data Exploration:
- Visualized the distribution of house prices using a KDE plot.
- Analyzed the correlation of numerical features with the target variable (`price`).
- Plotted scatter plots to examine relationships between `price` and features such as `sqft_living`, `long`, and `lat`.
- Investigated the impact of features like `waterfront` and `bedrooms` on house prices.

### Outlier Handling:
- Identified the top 1% most expensive houses and visualized their geographical distribution using `lat` and `long`.

### Feature Scaling:
- Applied Min-Max scaling to normalize the features before feeding them into the neural network model.

## Model Development

### Neural Network Architecture:
- Used a **Sequential** model from Keras with the following layers:
  - Four hidden layers, each with 19 neurons and ReLU activation functions.
  - Output layer with a single neuron for predicting the price.
  
### Model Compilation:
- Compiled the model using the **Adam optimizer** and **Mean Squared Error (MSE)** loss function.
  
### Model Training:
- Trained the model using 400 epochs and a batch size of 128, with validation on the test set during training.

### Hyperparameters:
- Optimized the model using default settings, focusing on architecture and batch size.

## Evaluation Metrics

### Loss Curve:
- Plotted the training and validation loss curves to check for overfitting or underfitting.

### Performance Metrics:
- Evaluated the model using **mean absolute error (MAE)**, **mean squared error (MSE)**, and **root mean squared error (RMSE)**.
- Also computed the **explained variance score** to evaluate the model's explanatory power.

### Predictions:
- The model's predictions were compared to the actual house prices to assess its accuracy.
- A scatter plot was created to visualize the difference between actual and predicted values.

## Results

- The model's performance showed a reasonable fit with an acceptable level of error, but further tuning (e.g., architecture adjustments) could improve accuracy.
- The model successfully predicted house prices for a given set of features, as shown in the evaluation metrics and visualizations.

## Conclusion
This project demonstrated the use of a neural network for house price prediction. Despite some room for improvement in model accuracy, the project highlighted the importance of feature engineering, scaling, and model evaluation in predictive modeling. Further model tuning could enhance the results, but the neural network provided a good baseline for the prediction task.
