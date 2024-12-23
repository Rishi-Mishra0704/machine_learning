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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Dont worry about the warnings. it will work just fine
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report,confusion_matrix

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', "cancer_classification_nn.csv")

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

df.describe().transpose()

sns.countplot(x="benign_0__mal_1",data=df)

df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")

X = df.drop("benign_0__mal_1",axis=1).values
y = df["benign_0__mal_1"]. values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape

model = Sequential()
model.add(Dense(30,activation="relu"))
model.add(Dense(15,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss=BinaryCrossentropy())

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test) ,epochs=600,verbose=2)

loss_df = pd.DataFrame(model.history.history)

loss_df.plot()

model = Sequential()
model.add(Dense(30,activation="relu"))
model.add(Dense(15,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss=BinaryCrossentropy())

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1,patience=25)

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=600,verbose=2, callbacks=[early_stop])

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

model = Sequential()
model.add(Dense(30,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(15,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss=BinaryCrossentropy())

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=600,verbose=2, callbacks=[early_stop])

plt.figure(figsize=(10,6))
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

predictions = (model.predict(X_test) > 0.5)*1 
predictions

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))
