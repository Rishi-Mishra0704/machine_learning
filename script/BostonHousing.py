import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/BostonHousing.csv')
df.head()

#cleaning data
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()

#get summary statistics
df_cleaned.describe()

# Visualizing the distributions of numerical features
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_cleaned.columns):
    plt.subplot(3, 5, i+1)
    sns.histplot(df_cleaned[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Visualizing the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


