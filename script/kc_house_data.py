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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', "kc_house_data.csv")

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

df.isnull().sum()

df.describe().transpose()

sns.displot(data=df["price"],kde=True, height=4, aspect=2)
plt.xlabel("Price(in million)")

df_nums = df.select_dtypes(include=["number"])
df_nums.corr()["price"].sort_values()

plt.figure(figsize=(10,6))
sns.scatterplot(x="price",y="sqft_living",data=df)
plt.xlabel("Price(in million)")

plt.figure(figsize=(10,6))
sns.boxplot(x="bedrooms",y="price",data=df)
plt.ylabel("Price(in million)")

plt.figure(figsize=(12,8))
sns.scatterplot(x="price",y="long",data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x="price",y="lat",data=df)

plt.figure(figsize=(12,8))
sns.scatterplot(x="long",y="lat",data=df,hue="price")

df.sort_values("price",ascending=False).head(20)

len(df)*0.01

bottom_99_perc = df.sort_values("price",ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x="long",y="lat",data=bottom_99_perc,edgecolor=None,hue="price",palette="RdYlGn")

sns.boxplot(x="waterfront",y="price",data=df)

df.head()

df = df.drop("id",axis=1)

df["date"] = pd.to_datetime(df["date"])

df["date"]

df["year"]=df["date"].apply(lambda date: date.year)
df["month"]=df["date"].apply(lambda date: date.month)

df.head()

sns.boxplot(x="month",y="price",data=df,palette="coolwarm")

df.groupby("month").mean()["price"].plot()

df.groupby("year").mean()["price"].plot()

df = df.drop("date",axis=1)
df.head()

df["zipcode"].value_counts()

df = df.drop("zipcode",axis=1)
df.head()

X = df.drop("price",axis=1).values
y = df["price"]. values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# +
model = Sequential()
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))
model.add(Dense(19,activation="relu"))

model.add(Dense(1))
model.compile(optimizer="adam",loss=MeanSquaredError())
# -

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=128,epochs=400,verbose=2)

loss_df = pd.DataFrame(model.history.history)

loss_df.plot()

model.evaluate(X_train,y_train,verbose=0)

test_preds = model.predict(X_test)

test_preds

print(mean_absolute_error(y_test,test_preds))
print(mean_squared_error(y_test,test_preds))
print(np.sqrt(mean_squared_error(y_test,test_preds)))


explained_variance_score(y_test,test_preds)

plt.figure(figsize=(12,6))
plt.scatter(y_test,test_preds)
plt.plot(y_test,y_test,"r")

single_house = df.drop("price",axis=1).iloc[0]

single_house = scaler.transform(single_house.values.reshape(-1,19))

model.predict(single_house)

df.head(1)
