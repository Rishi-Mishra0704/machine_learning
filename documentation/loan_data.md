# Random Forest Project

## Objective
The goal of this project is to predict whether a borrower will pay back their loan in full based on a variety of financial and demographic features using machine learning.

## Dataset
The dataset consists of information about loans, including features like:
- **credit.policy**: Whether the borrower meets LendingClub's underwriting criteria.
- **purpose**: The purpose of the loan (e.g., debt consolidation, credit card, etc.).
- **int.rate**: The interest rate of the loan.
- **installment**: Monthly payment for the loan.
- **log.annual.inc**: The natural log of the borrower's reported annual income.
- **dti**: Debt-to-income ratio.
- **fico**: A credit score used by LendingClub.
- **days.with.cr.line**: The number of days the borrower has had a credit line.
- **revol.bal**: Revolving balance (amount unpaid at the end of the billing cycle).
- **revol.util**: Revolving line utilization rate.
- **inq.last.6mths**: The borrower's number of credit inquiries in the last 6 months.
- **delinq.2yrs**: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- **pub.rec**: The borrower's number of derogatory public records (bankruptcies, tax liens, etc.).

## Exploratory Data Analysis (EDA)
Several visualizations were created to explore the data, including:
- **FICO score distribution**: Plotted by credit policy and whether the loan was paid in full.
- **Loan purpose vs. loan payment status**: A count plot showing loan purposes with respect to payment status.
- **FICO score vs. interest rate**: To check the relationship between a borrower's credit score and their interest rate.
- **Interest rate vs. FICO**: The `lmplot` shows the relationship between these two features, split by `not.fully.paid` and `credit.policy`.

## Data Preprocessing
- Categorical variables like `purpose` were one-hot encoded.
- Features like `fico` and `int.rate` were explored through visualizations.
- The data was split into training and test sets.

## Model Training
Two models were trained:
1. **Decision Tree Classifier**: The model was trained and evaluated with predictions on the test set. A classification report and confusion matrix were generated.
2. **Random Forest Classifier**: This model was trained and evaluated in a similar manner.

## Results and Evaluation
- Both models were evaluated using classification reports and confusion matrices.
- **Random Forest** performed slightly better in some areas, but **Decision Tree** had better recall in certain cases.

## Conclusion
Both models are useful for predicting loan repayment, but Random Forest provided a more balanced performance overall. However, the Decision Tree performed better on recall in some instances.
