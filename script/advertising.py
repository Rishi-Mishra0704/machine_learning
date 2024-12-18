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
# # Logistic Regression Project 
#
# In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
#
# This data set contains the following features:
#
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
#
# ## Import Libraries
#
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# +
# Get the absolute path of the current working directory
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'advertising.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

# ** Use info and describe() on ad_data**

df.info()

df.describe()

# ## Exploratory Data Analysis
#
# Let's use seaborn to explore the data!
#
# Try recreating the plots shown below!
#
# ** Create a histogram of the Age**

sns.histplot(data=df,x=df["Age"],bins=30)

# **Create a jointplot showing Area Income versus Age.**

sns.jointplot(data=df, x="Age", y="Area Income")

# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

sns.jointplot(x="Age",y="Daily Time Spent on Site", data=df,kind="kde")

# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

sns.jointplot(data=df, x="Daily Time Spent on Site", y="Daily Internet Usage")

# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

sns.pairplot(df,hue="Clicked on Ad")

# # Logistic Regression
#
# Now it's time to do a train test split, and train our model!
#
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

X = df[["Daily Time Spent on Site", "Daily Internet Usage", "Age","Area Income","Male" ]]
y = df["Clicked on Ad"]
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=101)

# ** Train and fit a logistic regression model on the training set.**

model = LogisticRegression(solver="liblinear")
model.fit(X_train,y_train)

# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

preds = model.predict(X_test)

# ** Create a classification report for the model.**

print(metrics.classification_report(y_test,preds))
print(metrics.confusion_matrix(y_test,preds))


# ## Great Job!
