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
from sklearn.metrics import mean_absolute_error,mean_squared_error

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
