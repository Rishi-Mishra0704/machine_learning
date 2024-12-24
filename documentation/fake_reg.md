# Fake Regression Problem with Keras

## Objective
The goal of this project is to demonstrate how to use the Keras library for a regression task. In this case, the dataset is synthetic, and we are predicting a target variable (price) based on two features (feature1 and feature2).

## Dataset
The dataset used for this project is a CSV file named `fake_reg.csv`, which contains the following columns:

- **price**: The target variable representing the price of an item.
- **feature1**: A numeric feature used for predicting the price.
- **feature2**: Another numeric feature used for predicting the price.

Example data:
| price     | feature1   | feature2   |
|-----------|------------|------------|
| 461.53    | 999.79     | 999.77     |
| 548.13    | 998.86     | 1001.04    |
| 410.30    | 1000.07    | 998.84     |
| 540.38    | 999.95     | 1000.44    |
| 546.02    | 1000.45    | 1000.34    |

## Data Preprocessing

### Data Loading:
- The dataset is loaded from a CSV file using pandas and examined using `df.head()`.

### Feature Scaling:
- The features (`feature1` and `feature2`) are scaled using **MinMaxScaler** to ensure that the model trains more effectively. The scaling is applied separately to the training and test sets.

### Data Splitting:
- The dataset is split into training (70%) and testing (30%) sets using `train_test_split` from scikit-learn.

## Model Development

### Model Architecture:
- A **Sequential** neural network model is created using the Keras library, consisting of three hidden layers and one output layer:
  - **Layer 1**: Dense layer with 4 neurons and ReLU activation.
  - **Layer 2**: Dense layer with 4 neurons and ReLU activation.
  - **Layer 3**: Dense layer with 4 neurons and ReLU activation.
  - **Output Layer**: Dense layer with 1 neuron for the price prediction.

### Model Compilation:
- The model is compiled with the **RMSprop** optimizer and **Mean Squared Error** (MSE) as the loss function.

### Model Training:
- The model is trained for 250 epochs using the training data, with verbose set to 2 for progress tracking.

### Model Evaluation:
- The model's performance is evaluated using the training data, and a loss plot is generated to visualize the training process.

### Model Prediction:
- The model's predictions on the test set are compared against the true values using metrics such as **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.
- A scatter plot is generated to visualize the predicted values against the actual values.

## Model Saving and Reloading:
- The trained model is saved to a file (`my_gem_price_predictor.h5`) using the `save()` method.
- The saved model is later reloaded and used for prediction on new data.

## Evaluation Metrics:
- **Mean Absolute Error (MAE)** is calculated to measure the average magnitude of the errors in predictions.
- **Mean Squared Error (MSE)** is computed to measure the average squared difference between the predicted and actual values.

## Results

### Model Performance:
- Predictions on the test set show how well the model has learned to predict prices.
- A scatter plot comparing the true values of the test set against the model's predictions helps visualize the accuracy of the predictions.

### Evaluation Metrics:
- **MAE** and **MSE** are computed to evaluate the model's performance. Lower values indicate better model accuracy.

### Model Prediction on New Data:
- The model is used to predict prices for new data points (e.g., `[[998, 1000]]`), and the predictions are displayed.

## Conclusion
This project demonstrates the use of the Keras library to build a simple neural network model for a regression problem. By training the model on synthetic data, we can predict the price based on two features. Evaluation metrics such as MAE and MSE help assess the model's performance, and the model can be saved and reloaded for future predictions.
