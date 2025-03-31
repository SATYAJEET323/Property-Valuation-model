# -*- coding: utf-8 -*-
"""Nlp task 1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v1_ztjLpjRivIarClYtL3911tyfmJZOb

Section 1: Load the Dataset
"""

import pandas as pd
from google.colab import files

# csv format "Input" , "Prediction
uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded)))
print("Dataset shape:", df.shape)
df.head()

from google.colab import sheets
sheet = sheets.InteractiveSheet(df=df)

"""Section 2: Exploratory Data Analysis (EDA)

check basic info, summary statistics, class distribution, and missing values.
"""

import matplotlib.pyplot as plt
import seaborn as sns

print(df.info())
print(df.describe())

# visualizing class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Prediction', data=df)
plt.title("Distribution of Online Course Categories")
plt.xticks(rotation=45)
plt.show()

# missing value check
print("Missing values in each column:\n", df.isnull().sum())

"""Section 3: Text Processing

clean the course description text from the Input column.
"""

import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # cut non alphabetic characters numbers newlines etc and convert them to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text.lower().strip()

# clean text form "Input" column
df['cleaned_text'] = df['Input'].apply(clean_text)
df[['Input', 'cleaned_text']].head()

"""Section 4: Generate NLP-Based Features
We derive simple numeric features from the cleaned text.

"""

import os
import joblib
import re
import numpy as np

def combination_features(text, tfidf_vectorizer):
    """Combine NLP features and TFIDF features for a given text."""
    text_cleaned = clean_text(text)
    temp_df = pd.DataFrame({'cleaned_text': [text_cleaned]})
    temp_df = generate_nlp_features(temp_df)

    from scipy.sparse import csr_matrix, hstack
    nlp_feats = temp_df[['char_count', 'word_count', 'avg_word_length', 'stopword_count', 'htag_count']].values
    nlp_sparse = csr_matrix(nlp_feats)

    tfidf_feats = tfidf_vectorizer.transform([text_cleaned])
    combined = hstack([nlp_sparse, tfidf_feats])
    print("Combined feature shape:", combined.shape)
    return combined

new_text = input("Enter an Property description to classify: ")
new_text_cleaned = clean_text(new_text)

new_text_features_bow = bow_vectorizer.transform([new_text_cleaned])
new_text_features_tfidf = tfidf_vectorizer.transform([new_text_cleaned])
new_text_features_combined = combination_features(new_text, tfidf_vectorizer)

model_dir = '/content/'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

#regex pattern
patterns = {
    'bow': re.compile(r'bow', re.IGNORECASE),
    'tfidf': re.compile(r'tfidf', re.IGNORECASE),
    'fasttext': re.compile(r'fasttext', re.IGNORECASE),
    'combined': re.compile(r'combined', re.IGNORECASE)
}

print("\n--- Model Predictions ---")
if not model_files:
    print("No model files found.")
else:
    for model_file in model_files:
        print(f"\nLoading model: {model_file}")
        loaded_model = joblib.load(os.path.join(model_dir, model_file))
        try:
            if patterns['bow'].search(model_file):
                prediction = loaded_model.predict(new_text_features_bow)
                print(f"Prediction using BoW model ({model_file}): {prediction}")
            elif patterns['tfidf'].search(model_file):
                prediction = loaded_model.predict(new_text_features_tfidf)
                print(f"Prediction using TFIDF model ({model_file}): {prediction}")
            elif patterns['combined'].search(model_file):
                prediction = loaded_model.predict(new_text_features_combined)
                print(f"Prediction using Combined (NLP + TFIDF) model ({model_file}): {prediction}")
            elif patterns['fasttext'].search(model_file):
                new_text_fasttext = np.array([get_fasttext_embeddings(new_text_cleaned)])
                prediction = loaded_model.predict(new_text_fasttext)
                print(f"Prediction using FastText model ({model_file}): {prediction}")
            else:
                print(f"Unknown or unsupported model type in file: {model_file}")
        except Exception as e:
            print(f"Error predicting with model {model_file}: {e}")

import numpy as np

def generate_nlp_features(df):
    # char count
    df['char_count'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['cleaned_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    df['stopword_count'] = df['cleaned_text'].apply(lambda x: len([word for word in x.split() if word in stop_words]))
    df['htag_count'] = df['cleaned_text'].apply(lambda x: x.count('#'))
    return df

df = generate_nlp_features(df)
df[['char_count', 'word_count', 'avg_word_length', 'stopword_count', 'htag_count']].head()

"""Section 5: Generate Text Embedding Features

features using Bag of Words, TFIDF, and FastText embeddings.
"""

# installs fasttext if not done already
!pip install fasttext

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import fasttext
import os

# downloads pretrained fasttext model
if not os.path.exists('cc.en.300.bin'):
    !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    !gunzip cc.en.300.bin.gz

# bog of words
bow_vectorizer = CountVectorizer(max_features=5000)
X_bow = bow_vectorizer.fit_transform(df['cleaned_text'])

# tfidf
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
# fasttext embeddings load pretrained fs model
ft_model = fasttext.load_model('cc.en.300.bin')

def get_fasttext_embeddings(text):
    words = text.split()
    if words:
        word_vectors = [ft_model.get_word_vector(word) for word in words]
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(ft_model.get_dimension())

df['fasttext_embeddings'] = df['cleaned_text'].apply(get_fasttext_embeddings)
# testing one rows embedding
print("FastText embedding shape:", df['fasttext_embeddings'].iloc[0].shape)

"""Section 6: Train Supervised Models on NLP-Based Features
We use the numeric features generated earlier for training classifiers
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# select nlp features  &target
X_nlp = df[['char_count', 'word_count', 'avg_word_length', 'stopword_count', 'htag_count']]
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X_nlp, y, test_size=0.2, random_state=42)

#using Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Random Forest Classifier Report (NLP Features):")
print(classification_report(y_test, y_pred))

"""Section 7: Train Models on BoW, TFIDF, and FastText Features

use define a helper function to train several classifiers and compare there performance
"""

!pip install tabulate

import joblib
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_name):
    models = {
        'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000),
        'SVM': SVC(kernel='linear'),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} on {feature_name} features...")
        if name == 'Naive Bayes' and feature_name == "fasttext":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train) # fits and transform training data
            X_test = scaler.transform(X_test) #transform test data using fittedd scaler

        model.fit(X_train, y_train)
        joblib.dump(model, f'{name}_{feature_name}_features.pkl')
        y_pred = model.predict(X_test)
        results[name] = classification_report(y_test, y_pred, output_dict=True)
        print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    # create &display  summary table
    summary = pd.DataFrame({model: results[model]['weighted avg'] for model in results}).T
    summary = summary[['precision', 'recall', 'f1-score']]
    print("\nModel Comparison for", feature_name, "features:")
    print(tabulate(summary, headers='keys', tablefmt='grid'))
    return results

# train on baw features
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y, test_size=0.2, random_state=42)
results_bow = train_and_evaluate(X_train_bow, X_test_bow, y_train_bow, y_test_bow, "bow")

#trains on tfidf feature
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
results_tfidf = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, "tfidf")

# fast text stack embeddings form 2D array
fasttext_embeddings = np.vstack(df['fasttext_embeddings'])
X_train_fasttext, X_test_fasttext, y_train_fasttext, y_test_fasttext = train_test_split(fasttext_embeddings, y, test_size=0.2, random_state=42)
results_fasttext = train_and_evaluate(X_train_fasttext, X_test_fasttext, y_train_fasttext, y_test_fasttext, "fasttext")

"""Section 8: Combined Features (NLP + TFIDF)"""

from scipy.sparse import hstack, csr_matrix

# combines nlp feature (with tfidf
nlp_sparse = csr_matrix(X_nlp.values)
X_combined = hstack([nlp_sparse, X_tfidf])

X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(X_combined, y, test_size=0.2, random_state=42)
results_combined = train_and_evaluate(X_train_comb, X_test_comb, y_train_comb, y_test_comb, "Combined_TFIDF")

"""Section 9: Save Models and Make Real-World Predictions

Section 10: Analysis of Results
After training and testing, compare the model performances. (For example, you might note that certain features or combinations perform better for categorizing online courses. Update your analysis as needed based on your dataset’s outcomes.)
"""

import pandas as pd
from tabulate import tabulate

#function to extract &format summary metrix
def extract_summary(results_dict):
    summary = {}
    for model, metrics in results_dict.items():
        accuracy = metrics.get('accuracy', 0)
        note = ""
        #flags if accuracy is exactly 1 bcoz of dataset issue overfit
        if accuracy == 1.0:
            note = "Potential data issue/overfitting"
        summary[model] = {
            "precision": round(metrics['weighted avg']['precision'], 3),
            "recall": round(metrics['weighted avg']['recall'], 3),
            "f1-score": round(metrics['weighted avg']['f1-score'], 3),
            "accuracy": round(accuracy, 3),
            "note": note
        }
    return pd.DataFrame(summary).T

#results_bow,results_tfidf,results_fasttext,results_combined
df_bow = extract_summary(results_bow) if 'results_bow' in globals() else pd.DataFrame()
df_tfidf = extract_summary(results_tfidf) if 'results_tfidf' in globals() else pd.DataFrame()
df_fasttext = extract_summary(results_fasttext) if 'results_fasttext' in globals() else pd.DataFrame()
df_combined = extract_summary(results_combined) if 'results_combined' in globals() else pd.DataFrame()

print("Analysis of Model Performance\n")

if not df_bow.empty:
    print(" Bag of Words Features")
    print(tabulate(df_bow, headers='keys', tablefmt='grid'))
    print("\n")

if not df_tfidf.empty:
    print(" TFIDF Features")
    print(tabulate(df_tfidf, headers='keys', tablefmt='grid'))
    print("\n")

if not df_fasttext.empty:
    print(" FastText Features")
    print(tabulate(df_fasttext, headers='keys', tablefmt='grid'))
    print("\n")

if not df_combined.empty:
    print("Combined (NLP + TFIDF) Features")
    print(tabulate(df_combined, headers='keys', tablefmt='grid'))
    print("\n")

print("Overall Analysis:")
print("""
- Data Quality Check: If any model shows an accuracy of 1.0, this may indicate that the dataset is too easy, overfitting has occurred, or the labels are not properly distributed. Re-examine the dataset for potential errors.
- Model Performance:
   - Models trained on Bag of Words and TFIDF features generally demonstrate robust performance.
   - FastText-based models, while promising, show variability across classifiers, suggesting further hyperparameter tuning or more training data may be needed.
   - The combined feature approach (NLP + TFIDF) enriches the feature set but does not guarantee uniformly high performance across all models.
""")