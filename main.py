"""
Aspect-Based Sentiment Analysis in English and Turkish (to be provided for the latter language as well soon.)

This module performs aspect-based sentiment analysis using LDA for aspect extraction,
sentence-transformers for embeddings, and SVM for sentiment classification.

Author: Cem Rifki Aydin

"""

from collections import defaultdict
import re

import constants

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import spacy

import gensim
from gensim import corpora
from nltk.corpus import stopwords

# Attempt to download the model (only if not already installed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# Load sentence-transformers model
# ------------------------------
model = SentenceTransformer(constants.SBERT_MODEL_NAME).to(device)

en_stopwords = set(stopwords.words('english'))


def remove_stopwords(text):
    text = [word for word in text if word not in en_stopwords]
    return text


def csv_reader(input_path):

    df = pd.read_csv(input_path)

    # Drop rows where any value is NaN
    df = df.dropna(subset=["y"])  # .head(200)


    df["label"] = df["y"] 
    return df


# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def best_aspect(text, corpus_aspects, aspect_embs):
    a = []
    text_emb = model.encode(text)   
    # aspect_embs = [model.encode(asp) for asp in aspects] 
    for aspect_emb in aspect_embs:
        a.append(cosine_similarity(text_emb, aspect_emb))

    return corpus_aspects[np.argmax(a)]


# ------------------------------
# Generate embeddings
# ------------------------------

def embed_text_aspect(row):
    combined = f"{row['clean_review']}"
    return model.encode(combined)

# Preprocess with spaCy
def preprocess(text):
    doc = nlp(text)
    lst = [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and token.is_alpha and not token.is_stop]
    lst = remove_stopwords(lst)
    return lst

def generate_LDA_topic_aspects(docs, aspect_count):

    tokenized_docs = [preprocess(doc) for doc in docs]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Train LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10)

    # Print topics
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}:", topic)
    
    topics = [top[1] for top in lda_model.print_topics()]

    # Dictionary to store total weights
    token_weights = defaultdict(float)

    # Regex to extract weight-token pairs
    pattern = re.compile(r'([\d.]+)\s*\*\s*"([^"]+)"')

    for topic in topics:
        matches = pattern.findall(topic)
        for weight_str, token in matches:
            token_weights[token] += float(weight_str)

    # Sort by weight descending and get top 5
    top_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:aspect_count]

    aspects = []
    print("Top 5 tokens with highest total weight across all topics:")

    for token, weight in top_tokens:
        print(f"{token}: {weight:.3f}")
        aspects.append(token)

    return aspects

def generate_aspects_and_embeddings(df):
    # Sample English reviews
    docs = df["clean_review"].values

    corpus_aspects = generate_LDA_topic_aspects(docs, constants.ASPECT_COUNT)
    aspect_embeddings = [model.encode(asp) for asp in corpus_aspects]


    best_aspects = []

    for doc in docs:
        # In order to refrain from computing the embeddings of aspects over and over, I specify three inputs to the below function
        best_asp = best_aspect(doc, corpus_aspects, aspect_embeddings)
        best_aspects.append(best_asp)


    df["aspect"] = best_aspects
    return df

def main():
    df = csv_reader(constants.INPUT_PATH)
    df = generate_aspects_and_embeddings(df)

    # Split the DataFrame itself, preserving all columns
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    # Extract features and labels from the split DataFrames
    X_train = np.vstack(df_train.apply(embed_text_aspect, axis=1))
    y_train = df_train["label"].tolist()

    X_test = np.vstack(df_test.apply(embed_text_aspect, axis=1))
    y_test = df_test["label"].tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train, transform train
    X_test = scaler.transform(X_test)        # transform test

    clf = SVC(kernel='linear', C=1.0)  # You can also use 'rbf', 'poly', etc.
    clf.fit(X_train, y_train)

    # ------------------------------
    # Evaluation
    # ------------------------------
    # These are label mappings
    label_map = {"negative": 0, "positive": 1}

    # If you want to reverse it for reporting:
    inv_label_map = {v: k for k, v in label_map.items()}

    # Now use this dict to generate `target_names` in order
    target_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]


    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # The predicted aspects and their corresponding sentiments of the test data are written to another .csv file as shown below:
    pd.DataFrame({"text": df_test["clean_review"], "aspect": df_test["aspect"], "sentiment": y_pred}).to_csv("test_asp_sents.csv", index=False)

if __name__ == "__main__":
    main()