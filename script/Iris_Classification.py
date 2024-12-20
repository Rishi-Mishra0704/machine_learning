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

# ### Iris Classification

import os
import ssl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix


# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'iris.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

sns.pairplot(data=df,hue="species",palette="Dark2")

# +
import seaborn as sns
import matplotlib.pyplot as plt

# Filter the data for the species "setosa"
setosa = df[df["species"] == "setosa"]

# Create a bivariate kernel density plot
sns.kdeplot(
    x=setosa["sepal_width"], 
    y=setosa["sepal_length"], 
    cmap="plasma", 
    fill=True,
    thresh=0.1
)


# -

X = df.drop("species",axis=1)
y=df["species"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

svc = SVC()
svc.fit(X_train,y_train)

preds = svc.predict(X_test)

print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))

param_grid = {"C": [0.1,1,10,100], "gamma":[1,0.1,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid, verbose=2)
grid.fit(X_train,y_train)

grid_preds = grid.predict(X_test)
print(classification_report(y_test,grid_preds))
print(confusion_matrix(y_test,grid_preds))
