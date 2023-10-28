import sys
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
# sys.path.append('.Anupa/Chatbot')
import os 
sys.path.append(os.path.join(os.getcwd(), 'Anupa/Chatbot'))
from Chatbot_002_Test import *
import numpy as np
import spacy
import threading
import nltk
from nltk.stem import WordNetLemmatizer
sys.path.append(os.path.join(os.getcwd(), 'Anupa/StoryRecommender'))
from Stories import scrape
from transformers import pipeline
sys.path.append(os.path.join(os.getcwd(), 'Anupa/Translator'))
from SinhalaToEnglish import *
sys.path.append(os.path.join(os.getcwd(), 'Anupa/ActivityRecommender'))
from ActivitySelector import *

global usefulLinks
global context
global result
global problem_flag
result = None
problem_flag = False
context = ''
def all_func(text,mode):
    global t,y,result
    # Load the pre-trained word embeddings model
    nlp = spacy.load("en_core_web_md")
    data = json.loads(open('Anupa/Chatbot/intents_modified.json').read())
    has_result = False  # Flag variable
    global problem_flag
    
    sentences=data
    is_Sinhala = mode    # Enable sinhala

    morning_inputs=["good morning"]
    afternoon_inputs=["good afternoon"]
    greeting_inputs=["hi", "hey", "is anyone there?","hi there", "hello", "hey there", "howdy", "hola", "bonjour", "konnichiwa", "guten tag", "ola"]
    night_inputs=["good night"]
    Activity_inputs=["what should i do","i dont know what to do"]
    evening_inputs=["good evening"]
    req_story_inputs=["yea can you look for something","can you look for something online"]

    morning_outputs=["Good morning. I hope you had a good night's sleep. How are you feeling today? "]
    afternoon_outputs=["Good afternoon. How is your day going?"]
    greeting_outputs=["Hello there. Tell me how are you feeling today?", "Hi there. What brings you here today?", "Hi there. How are you feeling today?", "Great to see you. How do you feel currently?", "Hello there. Glad to see you're back. What's going on in your world right now?"]
    night_outputs=["Good night. Get some proper sleep", "Good night. Sweet dreams."]
    evening_outputs=["Good evening. How has your day been?"]
    req_story_outputs=["Okay!!","Ok!! I'll look for something online"]
    Activity_outputs=["Doing one of this might help, why don't you give it a try :) \n"]

    lemma = nltk.stem.WordNetLemmatizer()
    global usefulLinks
    global result_lock
    result_lock = threading.Lock()
    global context
    
    # lemmatizing words as a part of pre-processing
    def perform_lemmatization(tokens):
        return [lemma.lemmatize(token) for token in tokens]


    # removing punctuation
    remove_punctuation = dict((ord(punc), None) for punc in string.punctuation)

    def capture_context(string):
        keywords = ["bullying", "stress", "work stress", "money issues", "time management"] 

        # Lemmatize the keywords
        lemmatized_keywords = [perform_lemmatization(keyword.split()) for keyword in keywords]

        # Convert the string to lowercase for case-insensitive matching
        lowercase_string = string.lower()

        # Search for keywords in the string
        for keyword, lemmatized_keyword in zip(keywords, lemmatized_keywords):
            if any(lem_word in lowercase_string for lem_word in lemmatized_keyword):
                return keyword

        return None


    def is_problem(string):
        keywords = ["problem","issue","challenge","difficult","trouble","concern","obstacle","hurdle","struggle","dilemma","predicament","complication","frustration","difficulty","setback","barrier","conflict","trouble","grievance","discomfort","woe","distress","worry","anxiety","pain","dispute","complaint","hassle","plight","misfortune","disadvantage","stress"]
        lowercase_string = string.lower()
        for keyword in keywords:
            if keyword in lowercase_string:
                return True
        return False



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
        
    def generate_activity(str):
        if punc_remove(str.lower()) in Activity_inputs:
            return random.choice(Activity_outputs)

    def generate_resourcelink(str):
        if punc_remove(str.lower()) in req_story_inputs:
            return random.choice(req_story_outputs) 

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

    def OnlineLinks(Problem):
        global result
        resources = []
        resources = scrape(Problem)
        usefulLinks = resources
        with result_lock:
            result = usefulLinks

    def Activities(Problem):
        ActivitityAll = Selection(Problem)
        return ActivitityAll

            

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

    def printer(query):
        if is_Sinhala==True:
            to_print=translate_text(query)
            return to_print #COMMENT
        else:
            to_print=query
            return to_print #COMMENT
        # print(to_print)  #UNCOMMENT
    
    continue_chat = True
    # printer('Hi! I am HelperBot. You can ask me anything and I shall try my best to answer them :) ')
    # while continue_chat:
    if is_Sinhala==True:
        # user_input = input()
        user_input = text
        user_input = translate_text_siToEn(user_input)
        user_input = user_input.lower()
    else:    
        # user_input = input().lower()
        user_input = text.lower()
    user_input = punc_remove(user_input)
    problem_flag=is_problem(user_input)
    if problem_flag==True:
        context = capture_context(user_input) ## only capure context when there is a problem
        # print(context)
    problem_flag = False
    if user_input != 'bye':
        if user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you':
            continue_chat = False
            x=printer('HelperBot: Not a problem! (And you are WELCOME! :D)')
            # continue
        else:
            if generate_greeting_response(user_input) is not None:
                x=printer('HelperBot: ' + generate_greeting_response(user_input))
            elif generate_goodmorning_response(user_input) is not None:
                x=printer('HelperBot: ' + generate_goodmorning_response(user_input))
            elif generate_goodnight_response(user_input) is not None:
                x=printer('HelperBot: ' + generate_goodnight_response(user_input))   
            elif generate_activity(user_input) is not None:
                x=printer('HelperBot: ' + generate_activity(user_input))  
                p= Activities('stress')

                x += ', '.join(str(item) for item in p)

            elif generate_resourcelink(user_input) is not None:
                x=printer('HelperBot: ' + generate_resourcelink(user_input))
                download_thread = threading.Thread(target=OnlineLinks, name="Downloader", args=(context,))
                download_thread.start()
                # Retrieve the result
            else:
                print('HelperBot: ', end='')
                x=printer(generate_response(user_input))
                with result_lock:
                    if result is None:
                        pass                 
                    else:                
                        t= ""
                        y= None
                        y=printer('\nHelperBot: by the way reading one of these would be helpful...give it a try ;)\n')
                        for hyperlink in result:
                            t += f'\n<a href="{hyperlink}" target="_blank">{hyperlink}</a><br> \n'  # Format as clickable link
                        y += t
                        result = None
                        x += "\n\n" + y
    else:
        # continue_chat = False
        x = printer('HelperBot: Bye, take care, and stay safe! :)')
    return x