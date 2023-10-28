import random
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import bs4 as bs
import warnings
import urllib.request
import nltk
import random
import string
import re
import collections
import math
from collections import Counter
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
from keras.models import Sequential, load_model
import sys
sys.path.append('./Anupa/Chatbot')
import os

lemmetizer = WordNetLemmatizer()

intents = json.loads(open('./Anupa/Chatbot/intents_modified.json').read())

words = pickle.load(open('./Anupa/Chatbot/words.pkl','rb'))
classes = pickle.load(open('./Anupa/Chatbot/classes.pkl','rb'))
model = load_model('./Anupa/Chatbot/chatbot_002.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmetizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)


def predict_class(sentense):
    bow =bag_of_words(sentense)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD =0.25
    results = [[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result

print("bot running")

def run(message):
    ints = predict_class(message)
    res = get_response(ints,intents)
    return "HelperBot: " + res