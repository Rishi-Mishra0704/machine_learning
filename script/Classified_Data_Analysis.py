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

# ## K Nearest Neighbour(KNN) Classifier
#
# You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes. 
#
# We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.
#
# Let's grab it and use it!

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'Classified_data.csv')

# Load the CSV file
df = pd.read_csv(csv_path,index_col=0)
df.head()
# -

scaler = StandardScaler()

scaler.fit(df.drop("TARGET CLASS",  axis=1))

scaled_features = scaler.transform(df.drop("TARGET CLASS",axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()

X = df_feat
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=101)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

preds = knn.predict(X_test)

print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))

# ### Elbow method for correct K value

# +
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    preds_i = knn.predict(X_test)
    error_rate.append(np.mean(preds_i != y_test))


# -

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color="blue",linestyle="dashed",marker="o",markerfacecolor="red",markersize=10)
plt.title("Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Value")

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
print(confusion_matrix(y_test,preds))
print("\n")
print(classification_report(y_test,preds))


