# ### LINEAR REGRESSION BASICS
#

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# +
# Get the absolute path of the current working directory
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
csv_path = os.path.join(working_dir, "..", 'data', 'USA_Housing.csv')

# Load the CSV file
df = pd.read_csv(csv_path)
df.head()
# -

df.info()

df.describe()

sns.pairplot(df)

sns.displot(df["Price"], kde=True)

# +
# Drop the 'Address' column as it contains string values
cleaned_df = df.drop(columns=['Address'])

# Plot the heatmap of the correlation matrix
sns.heatmap(cleaned_df.corr(), annot=True, cmap='coolwarm')
# -

cleaned_df.columns

X = cleaned_df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population',]]

y = cleaned_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

model =  LinearRegression()
model.fit(X_train,y_train)

print(model.intercept_)

model.coef_

X_train.columns

cdf = pd.DataFrame(model.coef_, X.columns, columns=["coeff"])

cdf

predictions = model.predict(X_test)
predictions

y_test

plt.scatter(y_test,predictions)

sns.displot((y_test-predictions), kde=True)

metrics.mean_absolute_error(y_test,predictions)

metrics.mean_squared_error(y_test,predictions)

np.sqrt(metrics.mean_squared_error(y_test,predictions))

metrics.r2_score(y_test,predictions)
