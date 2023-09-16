import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.load(open().read)

words = pickle.load(open(), 'rb')
classes = pickle.load(open(), 'rb')
model = load_model('chatbot_model.model')


def clean_up_sentence(sentence):
    sentence_word_array = nltk.word_tokenize(sentence)
    sentence_words = lemmatizer.lemmatize(sentence_word_array)
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
            else:
                bag[i] = 0
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array(bow))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probabilities': str(r[1])})
    return return_list
