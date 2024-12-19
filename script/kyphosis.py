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
df = pd.read_csv(csv_path,index_col=0)
df.head()
# -

sns.pairplot(df,hue="Kyphosis")
