# Titanic Survival Prediction using Logistic Regression

## Objective
The goal of this project is to predict whether a passenger survived or not on the Titanic based on features such as age, class, sex, and fare. Logistic Regression is used as the classification model for this binary prediction task.

## Dataset
The dataset used is the **Titanic dataset**, which includes the following columns:
- **Survived**: Target variable indicating whether the passenger survived (1) or not (0).
- **Pclass**: The passenger's class (1, 2, or 3).
- **Name**: The name of the passenger.
- **Sex**: The gender of the passenger.
- **Age**: The age of the passenger.
- **SibSp**: The number of siblings or spouses aboard.
- **Parch**: The number of parents or children aboard.
- **Ticket**: The ticket number.
- **Fare**: The fare paid by the passenger.
- **Cabin**: The cabin number (many missing values).
- **Embarked**: The port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## Data Preprocessing

### Missing Values:
1. **Age**: Missing age values are imputed based on the passenger class (`Pclass`), with predefined median ages for each class.
2. **Cabin**: The `Cabin` column is dropped due to many missing values.
3. **Sex and Embarked**: One-hot encoding is applied to the `Sex` and `Embarked` columns to convert categorical variables into numerical ones.

### Feature Engineering:
- **Age Imputation**: The `Age` column is imputed based on the passenger's class, where Class 1 passengers are given an age of 37, Class 2 passengers an age of 29, and Class 3 passengers an age of 24.
- **One-Hot Encoding**: The `Sex` and `Embarked` columns are transformed into binary features using one-hot encoding.

### Data Splitting:
The dataset is split into feature variables (`X`) and the target variable (`y`). The data is then split into training (70%) and testing (30%) sets.

## Model Development

### Logistic Regression:
- **Model**: A logistic regression model with the "liblinear" solver is used for classification.
- **Training**: The model is trained on the training data (`X_train` and `y_train`).
- **Prediction**: The model makes predictions on the test data (`X_test`).

### Model Evaluation:
- **Classification Report**: The model’s performance is evaluated using precision, recall, and F1-score metrics.
- **Confusion Matrix**: The confusion matrix is printed to evaluate the number of true positives, false positives, true negatives, and false negatives.

## Results:
The logistic regression model provides classification metrics, which summarize the model's performance in terms of precision, recall, F1-score, and accuracy.

## Conclusion:
Logistic Regression is a suitable model for predicting Titanic survival based on the available features. Further improvements could include experimenting with different models, tuning hyperparameters, or adding more features.
