# SMS Spam Classification

## Objective
The objective of this project is to classify SMS messages as either "spam" or "ham" (non-spam) using natural language processing (NLP) techniques and machine learning. The classification is performed using a Naive Bayes classifier, and the text data is preprocessed with tokenization, stopword removal, and TF-IDF transformations.

## Dataset
The dataset used is the **SMS Spam Collection** dataset, which consists of SMS messages labeled as "spam" or "ham". The dataset includes the following columns:
- **label**: The category of the message ("spam" or "ham").
- **message**: The content of the SMS message.

### Data Preprocessing:
1. **Text Cleaning**: The following steps are performed for each message:
   - Removal of punctuation.
   - Removal of stopwords (common words such as "the", "and", etc.).
   
2. **Feature Engineering**: 
   - The length of each message is calculated and plotted to analyze the distribution of message lengths for both spam and ham messages.
   
3. **Text Vectorization**:
   - A **Bag of Words (BoW)** model is created using the `CountVectorizer`. This transforms each message into a vector based on word counts.
   - The **TF-IDF** (Term Frequency-Inverse Document Frequency) transformation is applied to the BoW vectors to scale the word frequencies.

## Model Development

### Model:
A **Naive Bayes classifier** (specifically `MultinomialNB`) is used to classify the messages into spam or ham.

- The classifier is trained using the transformed features (TF-IDF values).
- The pipeline includes three main steps:
  1. **CountVectorizer**: Tokenization and vectorization of the text data.
  2. **TfidfTransformer**: Transformation of the word frequency counts into TF-IDF values.
  3. **Multinomial Naive Bayes Classifier**: The model that predicts whether the message is spam or ham.

### Model Evaluation:
- The model is evaluated using **classification report**, which includes metrics like precision, recall, and F1-score.
- A **confusion matrix** is used to further evaluate the model's performance.

### Training and Testing:
- The dataset is split into training and testing sets (70% training and 30% testing).
- The model is trained on the training data and evaluated on the test data.

### Results:
The classifier's performance is summarized in the classification report, which gives a detailed view of how well the model is distinguishing between spam and ham messages.

## Conclusion
The Naive Bayes classifier successfully classifies SMS messages as either spam or ham based on the preprocessed text data. The use of TF-IDF for feature transformation improved the model's ability to differentiate between the two classes. The model shows good classification performance, and further improvements could involve experimenting with other classifiers or fine-tuning the model parameters.
