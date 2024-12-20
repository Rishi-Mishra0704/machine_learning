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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# +
working_dir = os.getcwd()

# Construct the path to the 'data' directory relative to the working directory
data_dir = os.path.join(working_dir, "..", "data")

# Load the 'u.data' CSV file
columns_names = ["user_id", "item_id", "rating", "timestamp"]
u_data_path = os.path.join(data_dir, "u.data.txt")
df= pd.read_csv(u_data_path, sep="\t", names=columns_names)
df.head()
# -

movie_title_path = os.path.join(data_dir,"Movie_Id_Titles.csv")
movie_titles = pd.read_csv(movie_title_path)
movie_titles.head()

df = pd.merge(df,movie_titles,on="item_id")
df.head()

sns.set_style("white")


df.groupby("title")["rating"].mean().sort_values(ascending=False).head()

df.groupby("title")["rating"].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
ratings.head()

ratings["num of ratings"] = pd.DataFrame(df.groupby("title")["rating"].count())
ratings.head()

sns.histplot(data=ratings,bins=70,x=ratings["num of ratings"])

sns.histplot(data=ratings,bins=70,x=ratings["rating"])


sns.jointplot(x="rating",y="num of ratings",data=ratings,alpha=0.5)

moviemat = df.pivot_table(index="user_id",columns="title",values="rating")
moviemat.head()

ratings.sort_values("num of ratings",ascending=False).head(10)

star_war_user_ratings = moviemat["Star Wars (1977)"]
liarliar_user_ratings = moviemat["Liar Liar (1997)"]


star_war_user_ratings.head()

liarliar_user_ratings.head()

similiar_to_star_wars = moviemat.corrwith(star_war_user_ratings)

similiar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similiar_to_star_wars, columns=["Correlation"])
corr_starwars.dropna()

corr_starwars = corr_starwars.join(ratings["num of ratings"])
corr_starwars.head()

corr_starwars[corr_starwars["num of ratings"]>100].sort_values("Correlation",ascending=False).head()

corr_liarliar = pd.DataFrame(similiar_to_liarliar,columns=["Correlation"])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings["num of ratings"])
corr_liarliar[corr_liarliar["num of ratings"]>100].sort_values("Correlation",ascending=False).head()
