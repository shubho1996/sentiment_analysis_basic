# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:27:42 2021

@author: Shubhodeep
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib

lemmatizer = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_sent(words):
    lemmatized_sentence = []
    # Tag the words using NLTK POS Tagger
    tagged_words = nltk.pos_tag(words)
    
    # Convert the NLTK Tagged data to word net tag data
    tagged_words_wordnet = [(word, nltk_tag_to_wordnet_tag(tag)) for word, tag in tagged_words]
    
    #Create a list of lemmatized words
    for word, tag in tagged_words_wordnet:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence

def preprocess_text(text):
    
    #Tokenize
    words = word_tokenize(text)
    
    #Remove Short
    words = [w.lower() for w in words if len(w)>2]
     
    # print("\n\n1.####", words, "####\n\n")
    
    #Remove Stop words
    stop_words = stopwords.words('english')
    words = [w for w in words if w not in stop_words]
    
    # print("\n\n2.####", words, "####\n\n")
    
    #Lemmatization
    words = lemmatize_sent(words)
    
    #Load Pickle File
    vect = joblib.load("./vectorizer.pkl")
    model = joblib.load("./model.pkl")
    
    #convert to bag of words representation
    X = vect.transform([' '.join(words)])
    
    y = model.predict(X)
    print("\n\n1. y = ", y, "\n\n")
    #Predict
    return y
    
    