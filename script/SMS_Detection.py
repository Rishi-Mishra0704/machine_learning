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

from nltk.corpus import stopwords
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import string
# %matplotlib inline

# +
working_dir = os.getcwd()

# Construct the path to the CSV file relative to the working directory
data_path = os.path.join(working_dir, "..", 'data', 'SMSSpamCollection')

# Load the CSV file
messages = [line.strip() for line in open(data_path)]
# -

messages[50]

for mess_no, message in enumerate(messages[:10]):
    print(mess_no,message,"\n")

df = pd.read_csv(data_path,sep="\t",names=["label","message"])
df.head()

df.describe()

df.groupby("label").describe()

df["length"] = df["message"].apply(len)
df.head()

df["length"].plot.hist(bins=100)

df["length"].describe()

df[df["length"]==910]["message"].iloc[0]

df.hist(column="length",by="label",bins=60,figsize=(12,4))


def text_process(message):
    """
        1. remove punc
        2. remove stop words
        3. return list of clean text words
    """

    nopunc = [c for c in message if c not in string.punctuation]
    nopunc = "".join(nopunc)
    nopunc.split()
    return [word for word in nopunc.split() if word.lower not in stopwords.words("english")]



df.head()

df["message"].head().apply(text_process)

bow_transformer = CountVectorizer(analyzer=text_process).fit(df["message"])

print(len(bow_transformer.vocabulary_))

mess4 = df["message"][3]

bow4 = bow_transformer.transform([mess4])
print("bow 4:",bow4)
print("shape for bow 4",bow4.shape)

bow_transformer.get_feature_names_out()[9832]

messages_bow = bow_transformer.transform(df["message"])

print("Shape of sparse matrix", messages_bow.shape)

messages_bow.nnz

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))

tfidf = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf.transform(bow4)

print(tfidf4)

tfidf.idf_[bow_transformer.vocabulary_["university"]]

messages_tfidf = tfidf.transform(messages_bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf, df["label"])

spam_detect_model.predict(tfidf4)[0]

df["label"][3]

all_preds = spam_detect_model.predict(messages_tfidf)

all_preds

msg_train,msg_test,label_train,label_test = train_test_split(df["message"],df["label"],test_size=0.3)

pipeline = Pipeline([
    ("bow",CountVectorizer(analyzer=text_process)),
    ("tfidf", TfidfTransformer()),
    ("classifier",MultinomialNB())
])

pipeline.fit(msg_train,label_train)

preds = pipeline.predict(msg_test)

print(classification_report(label_test,preds))
