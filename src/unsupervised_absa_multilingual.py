"""
This module performs aspect-based sentiment analysis using LDA for aspect extraction,
sentence-transformers for embeddings, and SVM for sentiment classification.

Trains an unsupervised aspect-based sentiment analysis model using LDA for aspect extraction,
sentence-transformers for text embeddings, and SVM for sentiment classification.
Steps performed:
1. Loads language-specific models and stopwords.
2. Reads and preprocesses the input dataset.
3. Extracts aspects from the corpus using LDA topic modeling.
4. Assigns the most representative aspect to each document based on cosine similarity of embeddings.
5. Splits the data into training and test sets.
6. Generates sentence embeddings for each document.
7. Normalizes features and trains an SVM classifier for sentiment prediction.
8. Evaluates the classifier and prints a classification report.
9. Saves predicted aspects and sentiments for the test set to a CSV file.
10. Serializes the trained model, aspects, and embeddings to disk.
Args:
    args: Namespace or object containing configuration parameters:
        - lang: Language code ("en" or "tr").
        - dataset: Path to the input CSV file.
        - text: Column name containing text data.
        - aspect_count: Number of aspects to extract.
        - output_dir: Directory to save outputs.
Returns:
    pd.DataFrame: DataFrame containing the processed data with assigned aspects.
Aspect-Based Sentiment Analysis in English and Turkish 

Author: Cem Rifki Aydin
Date: 29.05.2025

"""

from collections import defaultdict
import pickle
import re
from typing import List

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import sys
sys.path.append("src")

from multiling_subclauses import SubclauseGenerator
import spacy

import gensim
from gensim import corpora
from nltk.corpus import stopwords

import os

import warnings
warnings.filterwarnings("ignore")


def prepare_pre_models(args):
    LANG = args.lang  # "en" for English, "tr" for Turkish
    LANG_MODEL = "tr_core_news_trf" if LANG == "tr" else "en_core_web_sm"  # Load the appropriate spaCy language model

    global nlp
    # Attempt to download the model (only if not already installed)
    try:
        nlp = spacy.load(LANG_MODEL)
    except OSError:
        print(f"Downloading {LANG_MODEL} model...")
        spacy.cli.download(LANG_MODEL)
        nlp = spacy.load(LANG_MODEL)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------
    # Load sentence-transformers model
    # ------------------------------
    SBERT_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr" if LANG == "tr" else "all-MiniLM-L6-v2"
    global sent_trans_model
    sent_trans_model = SentenceTransformer(SBERT_MODEL_NAME).to(device)

    global lang_stopwords
    # Load NLTK stopwords based on the language
    lang_stopwords = set(stopwords.words('turkish')) if LANG == "tr" else set(stopwords.words('english'))  

# Remove stopwords from texts in accordance with the list provided by the NLTK package.
def remove_stopwords(text):
    text = [word for word in text if word not in lang_stopwords]
    return text


# Read the csv file
def csv_reader(input_path):

    df = pd.read_csv(input_path)

    # Drop rows where any value corresponding to the sentiment column is NaN
    df = df.dropna(subset=["y"])  # .head(200)

    df["label"] = df["y"] 
    df = df.drop("y", axis=1)
    return df


# Cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------------------------------------------------------------------------------
# The most probable aspect is extracted from a text (e.g. a sentence or a review). 
# This relies on the already-created list of aspects from the whole corpus. 
# Sentence embeddings and the cosine similarity metric are leveraged.
# ------------------------------------------------------------------------------------------
def best_aspect(text, corpus_aspects, aspect_embs):
    a = []
    text_emb = sent_trans_model.encode(text)   
    
    for aspect_emb in aspect_embs:
        a.append(cosine_similarity(text_emb, aspect_emb))

    return corpus_aspects[np.argmax(a)]

# -----------------------------------------------------------------------------------
# The most similar subclause with respect to an aspect is found and returned. This is
# helpful in assigning the correct sentiment to this aspect in the end.
# -----------------------------------------------------------------------------------
def most_sim_subcl(aspect, subclauses):
    a = []
    text_emb = sent_trans_model.encode(aspect)   
    max_sim = -10
    most_sim_subcl_emb = None
    for subcl in subclauses:
        subcl_emb = sent_trans_model.encode(subcl)
        
        sim = cosine_similarity(text_emb, subcl_emb)
        if sim > max_sim:
            max_sim = sim
            most_sim_subcl_emb = subcl_emb

    return most_sim_subcl_emb


# --------------------
# Generate embeddings
# --------------------
def embed_text_aspect(row, args):
    combined = f"{row[args.text]}"
    return sent_trans_model.encode(combined)

# Preprocess with spaCy
def preprocess(text):
    doc = nlp(text)
    lst = [token.lemma_.lower() for token in doc if (token.pos_.lower() == "noun") and token.is_alpha and not token.is_stop]
    # If one only wants to eliminate stopwords defined with respect to the spaCy library, the below (NLTK) method can be commented out.
    lst = remove_stopwords(lst)
    return lst

# -----------------------------------------------------------------------------
# Extract the most representative aspects from the whole corpus (i.e., domain).
# The LDA model, which is an unsupervised approach, is accordingly leveraged.
# Top k (defined in args) aspects are obtained in the end.
# -----------------------------------------------------------------------------
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

    # Sort by weight descending and get top k
    top_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:aspect_count]

    aspects = []
    print(f"Top {aspect_count} tokens with highest total weight across all topics:")

    for token, weight in top_tokens:
        print(f"{token}: {weight:.4f}")
        aspects.append(token)

    return aspects


# Train the unsupervised model
def train(args=None):
    prepare_pre_models(args)
    # Sample reviews in English or Turkish
    df = csv_reader(args.dataset)


    docs = df.apply(lambda x: (x[args.text][0] if isinstance(x, pd.Series) else
                                          x[args.text][0] if isinstance(x, (list, np.ndarray)) else x), axis=1).values.tolist()

    corpus_aspects = generate_LDA_topic_aspects(docs, args.aspect_count)
    aspect_embeddings = [sent_trans_model.encode(asp) for asp in corpus_aspects]

    save_path = (args.output_dir or os.path.join("outputs", "unsupervised"))
    os.makedirs(save_path, exist_ok=True)

    best_aspects = []

    for doc in docs:
        # In order to refrain from computing the embeddings of aspects over and over, I specify three inputs to the below function
        best_asp = best_aspect(doc, corpus_aspects, aspect_embeddings)
        best_aspects.append(best_asp)

    print(len(best_aspects))
    df["aspect"] = best_aspects
    df["text"] = docs
    # Split the DataFrame itself, preserving all columns
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])

    # Extract features and labels from the split DataFrames
    X_train = np.vstack(df_train.apply(lambda x: embed_text_aspect(x, args), axis=1))
    y_train = df_train["label"].tolist()

    X_test = np.vstack(df_test.apply(lambda x: embed_text_aspect(x, args), axis=1))
    y_test = df_test["label"].tolist()

    # Normalize the sentence embedding features accordingly
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train, transform train
    X_test = scaler.transform(X_test)        # transform test
# 
    # The SVM classifier model is trained on the training data.
    clf = SVC(kernel='linear', random_state=42, C=1.0)  # You can also use 'rbf', 'poly', etc.
    clf.fit(X_train, y_train)


    # --------------------------------
    # Evaluation on the validation set
    # --------------------------------

    y_pred = clf.predict(X_test)

    # Suppose we have a mapping like this
    inv_label_map = {0: "negative", 1: "positive", 2: "neutral"}
    all_labels = sorted(set(y_train) | set(y_test) | set(y_pred))
    if args.lang == "en":
        y_test = [inv_label_map[i] for i in y_test]
        y_pred = [inv_label_map[i] for i in y_pred]
        target_names = [inv_label_map[i] for i in all_labels]
    else:
        target_names = all_labels

    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=target_names,
        target_names=target_names,
        zero_division=0  # avoids division by zero warnings
    ))

    # The predicted aspects and their corresponding sentiments of the test data are written to another .csv file as shown below:
    pd.DataFrame({"text": df_test[args.text].values.tolist(), "aspect": df_test["aspect"].values.tolist(), "sentiment": y_pred}).to_csv("test_asp_sents.csv", index=False)

    # --- Save everything
    with open(os.path.join(save_path, "unsuperv_model.pkl"), "wb") as f:
        pickle.dump({
            "corpus_aspects": corpus_aspects,
            "aspect_embeddings": aspect_embeddings,
            "clf": clf
        }, f)

    return df

def predict(args):
    """
    Load precomputed aspects + embeddings from pickle and 
    assign best aspect for each new document.
    """
    prepare_pre_models(args)
    sc = SubclauseGenerator(args.lang)
    load_path=os.path.join(args.model_dir, "unsuperv_model.pkl")

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No pickle file found at {load_path}. Run training first.")

    with open(load_path, "rb") as f:
        saved = pickle.load(f)

    corpus_aspects = saved["corpus_aspects"]
    aspect_embeddings = saved["aspect_embeddings"]
    clf = saved["clf"]

    # Assign best aspect for each new document
    docs: List[str] = args.text if isinstance(args.text, list) else [args.text]
    best_aspects = []
    asp_sentiments = []
    X_test = []
    for doc in docs:
        best_asp = best_aspect(doc, corpus_aspects, aspect_embeddings)
        best_aspects.append(best_asp)

        # print("The most representative aspects:", best_aspects)
        
        subclauses = [" ".join(subcl) for subcl in sc.convert_to_subclauses(doc)]
 
        # --- Predict sentiment

        most_sim_subcl_emb = most_sim_subcl(best_asp, subclauses)
        X_test.append(most_sim_subcl_emb)

    sentiments = clf.predict(X_test)

    print("The most representative aspects and their sentiments:", list(zip(best_aspects, sentiments)))
    return list(zip(best_aspects, asp_sentiments))

if __name__ == "__main__":
    pass