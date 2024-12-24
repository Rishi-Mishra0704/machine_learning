# Kyphosis Detection

## Objective
The objective of this project is to detect **Kyphosis**, a spinal disorder characterized by an excessive outward curvature of the spine, using machine learning algorithms. The goal is to predict whether a patient has Kyphosis based on features such as age, number of vertebrae involved, and the number of surgeries performed.

## Dataset
The dataset used for this project is `kyphosis.csv` and contains the following columns:
- **Kyphosis**: Target variable indicating whether the patient has Kyphosis (Yes/No).
- **Age**: Age of the patient.
- **Number**: The number of vertebrae involved.
- **Start**: The number of surgeries performed.

## Data Preprocessing

### Cleaning:
- Loaded the data from the CSV file.
- Checked for missing values and handled them appropriately.

### Feature Selection:
- The features include the patient's **Age**, **Number** of vertebrae involved, and **Start** (number of surgeries performed).
- The target variable is **Kyphosis**, which indicates the presence of the condition.

### Data Exploration:
- Visualized the data using a pairplot to examine the relationships between features and the target variable.
- Observed that **Kyphosis** is a categorical variable with two possible values (Yes/No).

## Model Development

### Train-Test Split:
- Split the data into training and test sets using a 70-30 split.

### Algorithms Used:
1. **Decision Tree Classifier**:
   - A decision tree classifier was used as a baseline model to classify patients based on the features provided.
   
2. **Random Forest Classifier**:
   - A random forest classifier with 200 estimators was used to improve model performance by leveraging multiple decision trees.

### Model Training:
- Both models were trained on the training set and evaluated on the test set.

## Evaluation Metrics

### Performance Metrics:
- The models were evaluated using **classification report** and **confusion matrix**.
- The classification report provides metrics like precision, recall, F1-score, and accuracy.
- The confusion matrix helps visualize the true positive, true negative, false positive, and false negative predictions.

## Results

- The **Decision Tree Classifier** provided an initial baseline performance.
- The **Random Forest Classifier** improved the results by averaging the predictions from multiple decision trees.
- The models' classification performance was evaluated, and the confusion matrix helped assess the models' strengths and weaknesses.

## Conclusion
This project demonstrated how to use decision tree-based algorithms, such as Decision Tree Classifier and Random Forest Classifier, for detecting **Kyphosis**. The random forest model provided better performance than the decision tree, indicating the benefits of using ensemble methods for classification tasks.
