# Iris Classification

## Objective
The goal of this project is to classify different species of the Iris flower based on four features: sepal length, sepal width, petal length, and petal width. The classification model will use a Support Vector Machine (SVM) to predict the species of the flower.

## Dataset
The dataset used for this project is a CSV file named `iris.csv`. It contains the following features:

- **sepal_length**: Length of the sepal.
- **sepal_width**: Width of the sepal.
- **petal_length**: Length of the petal.
- **petal_width**: Width of the petal.
- **species**: The species of the Iris flower (target variable), which can be one of three values: *setosa*, *versicolor*, or *virginica*.

## Data Preprocessing

### Cleaning:
- No missing values were identified in the dataset.
- The data was ready for model development without further cleaning.

### Feature and Target Split:
- Features (`X`) were extracted by dropping the `species` column.
- The target variable (`y`) was the `species` column.

### Data Splitting:
- The dataset was split into training (70%) and testing (30%) sets using `train_test_split`.

## Model Development

### SVM Model:
- A Support Vector Classifier (SVC) was instantiated and trained on the training set.
- The model was fitted using the training data and predictions were made on the test set.

### Hyperparameter Tuning:
- A **GridSearchCV** was used to optimize the model by performing an exhaustive search over a specified parameter grid for `C` and `gamma` values to find the best-performing model.

## Evaluation Metrics

### Classification Report:
- The **classification report** provides the precision, recall, and F1-score for each class, along with the overall accuracy.

### Confusion Matrix:
- The **confusion matrix** was used to evaluate the model's predictions by comparing the actual species against the predicted species.

## Results

### SVM Model:
- The initial SVM model gave a classification report with precision, recall, and F1-score for each species.

### GridSearchCV:
- The GridSearchCV optimized model provided better performance with the best hyperparameters for `C` and `gamma`.

## Conclusion
This project demonstrated the application of the Support Vector Machine (SVM) model for classification of the Iris flower species. The hyperparameter tuning with GridSearchCV helped in achieving better results by selecting optimal model parameters. The final model was evaluated using classification metrics such as accuracy, precision, recall, and confusion matrix.
