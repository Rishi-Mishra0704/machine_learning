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

# ## FAKE REGRESSION PROBLEM
# #### THIS IS A SIMPLE FAKE REGRESSION PROBLEM TO DEMONSTRATE THE USE OF THE KERAS LIBRARY

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Dont worry about the warnings. it will work just fine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error,mean_squared_error

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'fake_reg.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

sns.pairplot(df)

X = df.drop(["price"],axis=1).values
y = df["price"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# +
sequence_model = Sequential()
sequence_model.add(Dense(4,activation="relu"))
sequence_model.add(Dense(4,activation="relu"))
sequence_model.add(Dense(4,activation="relu"))
sequence_model.add(Dense(1))

sequence_model.compile(optimizer="rmsprop",loss=MeanSquaredError())

# +
#alternate way to instantiate Sequential Model
# sequence_model = Sequential([
# Dense(4,activation="relu"),
# Dense(4,activation="relu"),
# Dense(4,activation="relu"),
# Dense(1)
# ])
# -

sequence_model.fit(x=X_train,y=y_train,epochs=250,verbose=2)

loss_df = pd.DataFrame(sequence_model.history.history)

loss_df.plot()

sequence_model.evaluate(X_train,y_train,verbose=0)

test_preds = sequence_model.predict(X_test)

test_preds

test_preds = pd.Series(test_preds.reshape(300,))
pred_df = pd.DataFrame(y_test,columns=["Test True Y"])

pred_df = pd.concat([pred_df,test_preds],axis=1)
pred_df.columns = ["Test True Y", "Model Predictions"]

pred_df.head()

sns.scatterplot(x="Test True Y",y="Model Predictions", data=pred_df)

mean_absolute_error(pred_df["Test True Y"],pred_df["Model Predictions"])

mean_squared_error(pred_df["Test True Y"],pred_df["Model Predictions"])

new_gem=[[998,1000]]

new_gem = scaler.transform(new_gem)

sequence_model.predict(new_gem)

sequence_model.save("my_gem_price_predictor.h5")

later_model = load_model("my_gem_price_predictor.h5")
later_model.predict(new_gem)
