# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# ___
#
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Random Forest Project 
#
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
#
#
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
#
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
#
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

# %matplotlib inline

# ## Get the Data
#
# ** Use pandas to read loan_data.csv as a dataframe called loans.**

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'loan_data.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
# -

# ** Check out the info(), head(), and describe() methods on loans.**

df.info()

df.describe()

df.head()

# # Exploratory Data Analysis
#
#
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
#

# +
plt.figure(figsize=(10,6))
sns.histplot(df[df["credit.policy"] == 1]["fico"], bins=35, color="blue", label="Credit Policy = 1", kde=False, alpha=0.6)

# Plot for Credit Policy = 0
sns.histplot(df[df["credit.policy"] == 0]["fico"], bins=35, color="red", label="Credit Policy = 0", kde=False, alpha=0.6)

plt.legend()
plt.xlabel("FICO")
plt.show()
# -

# ** Create a similar figure, except this time select by the not.fully.paid column.**

# +
plt.figure(figsize=(10,6))
sns.histplot(df[df["not.fully.paid"] == 1]["fico"], bins=35, color="blue", label="Not Fully Paid = 1", kde=False, alpha=0.6)

sns.histplot(df[df["not.fully.paid"] == 0]["fico"], bins=35, color="red", label="Not Fully Paid = 0", kde=False, alpha=0.6)

plt.legend()
plt.xlabel("FICO")
plt.show()
# -

# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

plt.figure(figsize=(11,7))
sns.countplot(df,x="purpose", hue="not.fully.paid",palette="Set1")

sns.jointplot(x="fico",y="int.rate",data=df,color="purple")

# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**

plt.figure(figsize=(11,7))
sns.lmplot(y="int.rate",x="fico",data=df, hue="credit.policy",col="not.fully.paid",palette="Set1")

# # Setting up the Data
#
# Let's get ready to set up our data for our Random Forest Classification Model!
#

df.info()

# ## Categorical Features
#

category_feats = ["purpose"]
final_data = pd.get_dummies(df,columns=category_feats,drop_first=True)

# ## Train Test Split
#
# Now its time to split our data into a training set and a testing set!
#
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**

X = final_data.drop("not.fully.paid",axis=1)
y=final_data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# ## Training a Decision Tree Model
#

# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

preds = dtree.predict(X_test)

print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))

# ## Training the Random Forest model
#
# Now its time to train our model!
#
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**

rforest = RandomForestClassifier()

rforest.fit(X_train,y_train)

# ## Predictions and Evaluation
#
# Let's predict off the y_test values and evaluate our model.
#
# ** Predict the class of not.fully.paid for the X_test data.**

rfc_preds = rforest.predict(X_test)

# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**

print(classification_report(y_test,rfc_preds))

# **Show the Confusion Matrix for the predictions.**

print(confusion_matrix(y_test,rfc_preds))

# **What performed better the random forest or the decision tree?**

# Random Forest Performed slightly better than Decision Tree in some cases however the decision tree performed better in some cases like recall.Please refer the classification report for more details.

# # Great Job!
