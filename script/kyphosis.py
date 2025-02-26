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

# ### Kyphosis Detection
#
# **Kyphosis** is a spinal disorder characterized by an *excessive outward curvature* of the spine, leading to a **hunched** or **rounded back**.
#

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

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'kyphosis.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

sns.pairplot(df,hue="Kyphosis")


X = df.drop("Kyphosis",axis=1)
y = df["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

preds = dtree.predict(X_test)

print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))

rforest = RandomForestClassifier(n_estimators=200)
rforest.fit(X_train,y_train)

rtree_pred = rforest.predict(X_test)

print(classification_report(y_test,rtree_pred))
print(confusion_matrix(y_test,rtree_pred))
