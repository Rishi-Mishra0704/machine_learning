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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'Classified_data.csv')

# Load the CSV file
df = pd.read_csv(csv_path,index_col=0)
df.head()
