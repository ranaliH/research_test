import json
import sklearn
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
from Chatbot_002_Test import *
import numpy as np
import spacy
import pandas as pd

# Load the pre-trained word embeddings model
nlp = spacy.load("en_core_web_md")

with open('D:\SLIIT\Year 4\Semester 1\Research\ChatBot\Chatbot_dta\intents_modified.json', 'r') as f:
  data = json.load(f)

sentences=data

morning_inputs=["good morning"]
afternoon_inputs=["good afternoon"]
greeting_inputs=["hi", "hey", "is anyone there?","hi there", "hello", "hey there", "howdy", "hola", "bonjour", "konnichiwa", "guten tag", "ola"]
night_inputs=["good night"]
evening_inputs=["good evening"]

morning_outputs=["Good morning. I hope you had a good night's sleep. How are you feeling today? "]
afternoon_outputs=["Good afternoon. How is your day going?"]
greeting_outputs=["Hello there. Tell me how are you feeling today?", "Hi there. What brings you here today?", "Hi there. How are you feeling today?", "Great to see you. How do you feel currently?", "Hello there. Glad to see you're back. What's going on in your world right now?"]
night_outputs=["Good night. Get some proper sleep", "Good night. Sweet dreams."]
evening_outputs=["Good evening. How has your day been?"]

lemma = nltk.stem.WordNetLemmatizer()

df_results = pd.DataFrame(columns=['pattern', 'result', 'desired_response']) # where i will store the results
# lemmatizing words as a part of pre-processing
def perform_lemmatization(tokens):
    return [lemma.lemmatize(token) for token in tokens]


# removing punctuation
remove_punctuation = dict((ord(punc), None) for punc in string.punctuation)

# method to pre-process all the tokens utilizing the above functions
def processed_data(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(remove_punctuation)))

def punc_remove(str):
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ''

    for char in str:
        if char not in punctuations:
            no_punct = no_punct + char

    return no_punct

# method to generate a response to greetings
def generate_greeting_response(hello):
    if punc_remove(hello.lower()) in greeting_inputs:
        return random.choice(greeting_outputs)

# method to generate a response to conversations
def generate_goodmorning_response(str):
    if punc_remove(str.lower()) in morning_inputs:
        return random.choice(morning_outputs)
    

# method to generate a response to conversations
def generate_goodnight_response(str):
    if punc_remove(str.lower()) in night_inputs:
        return random.choice(night_outputs)
    

def cosvalue(s1, s2):
    # preprocess the input strings
    s1=s1.lower()
    s2=s2.lower()
    
    s1 = re.sub(r'[^\w\s]', '', s1).lower().split()
    s2 = re.sub(r'[^\w\s]', '', s2).lower().split()
    
    # create counters for the two strings
    count1 = Counter(s1)
    count2 = Counter(s2)
    
    # get the unique words from both strings
    words = set(count1.keys()).union(set(count2.keys()))
    
    # calculate the numerator and denominators for the cosine similarity formula
    dot_product = sum(count1.get(word, 0) * count2.get(word, 0) for word in words)
    magnitude1 = math.sqrt(sum(count1.get(word, 0)**2 for word in words))
    magnitude2 = math.sqrt(sum(count2.get(word, 0)**2 for word in words))
    
    # calculate the cosine similarity between the two strings
    if not magnitude1 or not magnitude2:
        return 0
    else:
        return dot_product / (magnitude1 * magnitude2)


def sentence_similarity(sentence1, sentence2):
    # Tokenize and obtain word embeddings for sentence 1
    doc1 = nlp(sentence1)
    tokens1 = [token.vector for token in doc1]

    # Tokenize and obtain word embeddings for sentence 2
    doc2 = nlp(sentence2)
    tokens2 = [token.vector for token in doc2]

    # Calculate the similarity between the sentence embeddings
    similarity_matrix = cosine_similarity(np.array(tokens1), np.array(tokens2))
    sentence_similarity = np.mean(similarity_matrix)

    return sentence_similarity


def generate_response(user):
    embeddings_dist=0
    bracrobo_response = ''
    res={}
    for intent in data['intents']:
        name=intent['tag']
        Scores = {name:[]}
        for response in intent['patterns']:
            temp=[response]
            temp.append(user)
            cosine=cosvalue(temp[0],temp[1])
            # edit_dist = edit_distance_percentage(temp[0],temp[1])
            Scores[name].append(cosine)

        res.update(Scores)

    max_val = None
    max_key = None

    for key, value in res.items():
        if max_val is None or max(value) > max_val:
            max_val = max(value)
            max_key = key
            # embeddings_dist=sentence_similarity(user,max_key)

    max_embeddings_dist=-1
    for intent in data['intents']:
            if intent['tag']==max_key:
                responses=intent['responses']
                patt=intent['patterns']  
                for paterns in patt: 
                    embeddings_dist=sentence_similarity(user,paterns)
                    if embeddings_dist>max_embeddings_dist:
                        max_embeddings_dist=embeddings_dist

    response = random.choice(responses)

    if max_val < 0.8 and max_embeddings_dist< 0.8:
        bracrobo_response = run(user)
        return bracrobo_response
    else:
        # responses = [resp for intent in data if intent['intents'] == max_key for resp in intent['responses']]
        # response = random.choice(responses)
        response
        for intent in data['intents']:
            if intent['tag']==max_key:
                responses=intent['responses']
                response = random.choice(responses)
        return response
    
    
continue_chat = True
print('Hi! I am HelperBot. You can ask me anything and I shall try my best to answer them :) ')
while continue_chat:
    user_input = input().lower()
    user_input = punc_remove(user_input)
    if user_input != 'bye':
        if user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you':
            continue_chat = False
            print('HelperBot: Not a problem! (And WELCOME! :D)')
#         elif user_input in convo_replies:
#             print('That\'s nice! How may I be of assistance?')
            continue
        else:
            if generate_greeting_response(user_input) is not None:
                print('HelperBot: ' + generate_greeting_response(user_input))
            elif generate_goodmorning_response(user_input) is not None:
                print('HelperBot: ' + generate_goodmorning_response(user_input))
            elif generate_goodnight_response(user_input) is not None:
                print('HelperBot: ' + generate_goodnight_response(user_input))
            else:
                print('HelperBot: ', end='')
                print(generate_response(user_input))
            print('-' * 100)
            print('Is the response correct? (y/n) ')
            evaluation = input().lower()
            if evaluation=='n':
                evaluation=0
                print('enter desired response ')
                desired_response=input().lower()
            else:
                evaluation = 1
                desired_response=""
            new_row = {'pattern':user_input, 'result':evaluation, 'desired_response':desired_response}
            df_results = pd.concat([df_results, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            print('-' * 100)
    else:
        continue_chat = False
        print('HelperBot: Bye, take care, and stay safe! :)')
        df_results.to_csv('../Accuracy/AccuracyChatbot_Hybrid_aftermodification.csv', mode='a', header=False, index=False)