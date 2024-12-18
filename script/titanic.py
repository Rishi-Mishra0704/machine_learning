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

# ## Logistic Regression Basics

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---
# Setup working directory and file paths
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data',)

# Load the CSV files
test_df = pd.read_csv(csv_path + '/titanic_test.csv')
train_df = pd.read_csv(csv_path + '/titanic_train.csv')

# ---
# Data Visualization (for exploration)
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train_df, hue="Pclass")
sns.displot(train_df['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=train_df)
train_df['Fare'].hist(bins=40, figsize=(10, 4))

# ---
# Data Cleanup - Handle missing values

# Visualize missing data
plt.figure(figsize=(10, 7))
sns.boxplot(x='Pclass', y='Age', data=train_df)

# Function to impute missing Age based on Pclass


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train_df["Age"] = train_df[["Age", "Pclass"]].apply(impute_age, axis=1)

# Dropping Cabin column
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

# One-hot encode Sex and Embarked columns
gender = pd.get_dummies(train_df["Sex"], drop_first=True).astype(int)
embark = pd.get_dummies(train_df["Embarked"], drop_first=True).astype(int)

# Concatenate the new columns to the original train_df
train_df = pd.concat([train_df, gender, embark], axis=1)

# Drop unnecessary columns from train_df
train_df.drop(["Sex", "Name", "Ticket", "Embarked",
              "PassengerId"], axis=1, inplace=True)

# ---
# Clean up test_df using the same steps as train_df
# Impute missing Age for test_df
test_df["Age"] = test_df[["Age", "Pclass"]].apply(impute_age, axis=1)

# One-hot encode Sex and Embarked columns for test_df
gender_test = pd.get_dummies(test_df["Sex"], drop_first=True).astype(int)
embark_test = pd.get_dummies(test_df["Embarked"], drop_first=True).astype(int)

# Concatenate the new columns to the original test_df
test_df = pd.concat([test_df, gender_test, embark_test], axis=1)

# Drop unnecessary columns from test_df
test_df.drop(["Sex", "Name", "Ticket", "Embarked",
             "PassengerId"], axis=1, inplace=True)

# ---
# Split train_df into X (features) and y (target variable)
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# ---
# Train the Logistic Regression model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# Predict on test set
preds = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, preds))
conf_matrix = confusion_matrix(y_test, preds)
print(conf_matrix)
