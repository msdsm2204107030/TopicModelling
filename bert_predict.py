import os
import re
import nltk
import torch
import joblib
import pickle
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI 
from bertopic import BERTopic
from string import punctuation
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from os.path import dirname, join, realpath

nltk.download('wordnet')

print('libraries loaded')

class PredictTopicOutput(BaseModel):
    numberOfTopics: int
    Topic1: str
    Topic2: str
    Topic3: str
    probability: str

app = FastAPI(
    title="Bertopic model ",
    description="FastAPI service to predict topics using BERTopic model (powered by LLAMA 2).",
    version="0.1",
)

print("loading the llama-bert-topic model")
model = BERTopic.load("llama_model_dir")
print('llama loaded!')

# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):

    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
    text = "".join([c for c in text if c not in punctuation])

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    return text

@app.get("/predict-tweet")
def predict_topics(tweet: str):
    """
    A simple function that receive a content and predict the topic of the content.
    :param tweet:
    :return: prediction, probabilities
    """

    cleaned_tweet = text_cleaning(tweet)
    
    num_of_topics = 3
    similar_topics, similarity = model.find_topics(cleaned_tweet, top_n=num_of_topics)
    topics_name=pd.read_excel("topic_list_by_llama.xlsx")
    topic_dict = topics_name.set_index("Topic")["Name"].to_dict()

    predictTopicOutput = PredictTopicOutput(
    numberOfTopics=num_of_topics,
    Topic1=topic_dict[similar_topics[0]],
    Topic2=topic_dict[similar_topics[1]],
    Topic3=topic_dict[similar_topics[2]],
    probability=str(np.round(similarity, 2))
)

    return predictTopicOutput
    
   