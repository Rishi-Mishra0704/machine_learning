# Linear Regression Project: Ecommerce Customer Analysis

## Problem Overview
The project focuses on analyzing customer data for an Ecommerce company that sells clothing both online and in-store. The company offers in-store style sessions and a mobile app experience for online shopping. They want to determine whether they should focus more on their mobile app or website development.

The dataset includes customer details like email, address, and avatar color, along with numerical data:
- **Avg. Session Length**: The average time customers spent in the store for style advice.
- **Time on App**: The average time customers spent on the app.
- **Time on Website**: The average time customers spent on the website.
- **Length of Membership**: The duration for which the customer has been a member.

The goal is to use Linear Regression to identify which factors influence yearly spending, helping the company decide whether to prioritize their mobile app or website.

## Dataset
The dataset, `Ecommerce_Customers.csv`, contains the following columns:
- **Email**: Customer's email address (categorical).
- **Address**: Customer's address (categorical).
- **Avatar Color**: Color associated with customer's avatar (categorical).
- **Avg. Session Length**: Average session duration for in-store style advice (numerical).
- **Time on App**: Average time spent by the customer on the mobile app in minutes (numerical).
- **Time on Website**: Average time spent by the customer on the website in minutes (numerical).
- **Length of Membership**: Duration for which the customer has been a member (numerical).
- **Yearly Amount Spent**: The target variable representing the amount spent by the customer in a year (numerical).

## 1. Data Loading and Initial Inspection
- The data is stored in a CSV file (`Ecommerce_Customers.csv`), which is read into a pandas DataFrame.
- **Initial checks** include displaying the first few rows of the data, along with a statistical summary (`describe()`) and a general overview (`info()`).

## 2. Exploratory Data Analysis (EDA)
- **Seaborn visualizations** are used to explore relationships between different features in the dataset:
  - **Jointplots** are used to visualize correlations between variables like:
    - Time on Website vs. Yearly Amount Spent
    - Time on App vs. Yearly Amount Spent
    - Time on App vs. Length of Membership
  - **Pairplot** visualizes relationships across all features to identify the most correlated features with the target variable, Yearly Amount Spent.

### Key Insights from EDA:
- `Length of Membership` shows the strongest correlation with `Yearly Amount Spent`.
- Time spent on the mobile app also has a noticeable impact on yearly spending.

## 3. Feature Selection and Data Splitting
- The dataset is split into features (independent variables) and the target variable (`Yearly Amount Spent`).
- The data is divided into training and testing sets (70% training, 30% testing).

## 4. Model Training
- A **Linear Regression model** is trained using the training data.
- After training, the model coefficients are printed, showing the impact of each feature on the target variable.

## 5. Model Evaluation
- Predictions are made using the test data, and a scatterplot is created to compare actual vs. predicted values.
- Model performance is evaluated using the following metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared**: A measure of how well the model fits the data.

## 6. Residuals Analysis
- The residuals (differences between predicted and actual values) are analyzed to ensure the model fits well. A histogram of the residuals is plotted, which should resemble a normal distribution.

## 7. Model Interpretation and Conclusion
- The coefficients of the model are examined to interpret the relationship between the features and the target variable.
- **Interpretation**:
  - `Length of Membership` is the most influential factor on yearly spending.
  - The mobile app, with a higher coefficient than the website, seems to have a stronger influence on spending.
  
  **Conclusion**: The Length of Membership has the most significant impact on yearly spending, followed by the time spent on the mobile app. Now there are 2 main options for the company:
    - Focus more on the mobile app development to drive more revenue.
    - Improve the website experience to catch up with the app's performance.

## Key Takeaways:
- Linear regression successfully identifies the most influential features on customer spending.
- The company should prioritize app development to drive more revenue.
