# Logistic Regression Project: Advertisement Click Prediction

## Objective
The aim of this project is to build a machine learning model using logistic regression to predict whether a user will click on an advertisement based on their behavior and demographic features.

## Dataset
The dataset contains user behavior and demographic data, with the following features:

- **Daily Time Spent on Site**: Time spent by the user on the website daily (in minutes).
- **Age**: User's age (in years).
- **Area Income**: Average income of the geographical area of the user.
- **Daily Internet Usage**: Average daily internet usage by the user (in minutes).
- **Ad Topic Line**: Text headline of the advertisement.
- **City**: User's city of residence.
- **Male**: Binary indicator of the user's gender (1 = Male, 0 = Female).
- **Country**: Country where the user is located.
- **Timestamp**: Date and time of the ad interaction.
- **Clicked on Ad**: Target variable indicating if the user clicked on the advertisement (1 = Yes, 0 = No).

## Exploratory Data Analysis (EDA)

### Visualizations
- **Age Distribution**: Histogram to visualize the distribution of user ages.
- **Area Income vs. Age**: Joint plot to examine the relationship between user age and area income.
- **Daily Time Spent on Site vs. Age**: Kernel density estimation plot to observe trends between age and site engagement.
- **Daily Time Spent on Site vs. Daily Internet Usage**: Joint plot to study the relationship between time spent on the site and overall internet usage.
- **Pairplot**: Visualized all feature relationships, with `Clicked on Ad` as the hue, to identify correlations.

### Observations
- Younger users tend to spend more time on the website.
- Area income and age show variability, but no strong linear correlation with ad clicks.
- Users with higher internet usage generally spend less time on the site.

## Model Development

### Train-Test Split
- **Features (X)**: Selected predictors:
  - `Daily Time Spent on Site`
  - `Age`
  - `Area Income`
  - `Daily Internet Usage`
  - `Male`
- **Target (y)**: `Clicked on Ad`.
- Split the data into training (70%) and testing (30%) sets using `train_test_split`.

### Model Training
- **Model**: Logistic Regression.
- Trained the model using the `liblinear` solver.

## Evaluation

### Predictions
- Predicted ad click probabilities on the test dataset.

### Metrics
- **Classification Report**:
  - Precision, recall, and F1-score were computed for both classes.
- **Confusion Matrix**:
  - Evaluated counts of true positives, true negatives, false positives, and false negatives.

### Results
- The model performed well, capturing patterns in user behavior with reasonable accuracy and precision.

## Visualizations
- **Feature Distribution**: Histograms and joint plots to explore feature characteristics.
- **Prediction Performance**: Confusion matrix and classification report to summarize model accuracy and error distribution.

## Conclusion
This project demonstrated the use of logistic regression to predict user ad clicks based on behavioral and demographic data. Insights gained from the EDA and model performance can help optimize advertisement targeting and enhance user engagement strategies.
