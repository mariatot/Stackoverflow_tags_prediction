import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import tensorflow as tf
import tensorflow_hub as hub
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

@st.cache_resource

def clean_html(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html5lib")
    for sent in soup(['style', 'script']):
        sent.decompose()
    return ' '.join(soup.stripped_strings)

def text_cleaning(text):
    import re
    pattern = re.compile(r'[^\w]|[\d_]')
    try: 
        res = re.sub(pattern," ", text).lower()
    except TypeError:
        return text
    pattern = re.compile('[^\\w\\s#]')
    try: 
        res = re.sub(pattern," ", res)
    except TypeError:
        return res
    pattern = re.compile(r'\w*\d+\w*')
    try: 
        res = re.sub(pattern," ", res)
    except TypeError:
        return res

    res = res.split(" ")        
    res = list(filter(lambda x: len(x)>2 , res))
    res = " ".join(res)
    return res

def tokenize(text):
    from nltk.corpus import stopwords
    from nltk import word_tokenize
    stop_words = set(stopwords.words('english'))
    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text
    res = [token for token in res if token not in stop_words]
    return res

def filtering_nouns(tokens):
    import nltk
    res = nltk.pos_tag(tokens)
    res = [token[0] for token in res if token[1] == 'NN']
    return res

def lemmatize(tokens):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token))  
    return lemmatized

def predict_tags(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = pickle.load(open('tfidf_model_fin2.pkl','rb'))
    res = vectorizer.transform([text])
        
    supervised_model = pickle.load(open('model_fin_supervised2.pkl','rb'))
    res = supervised_model.predict(res)
    
    mlb_model = pickle.load(open('mlb_model_fin2.pkl','rb'))
    res = mlb_model.inverse_transform(res)  
    return res
    

def main():
    st.title("Application de machine learning pour catégoriser automatiquement des questions")
    st.markdown("**OpenClassrooms** Projet n°5 du parcours Machine Learning")
    st.info("Auteur: Maria Filandrova")
    st.markdown("_"*10)
    
    if st.checkbox("Afficher le détail de la procédure", key=False):
            st.markdown("#### Prédiction de tags effectuée en utilisant un modèle \
                        de classification supervisée (regression logistique)")
    
    #cust_input = str(st.text_input("**Saisissez votre question**"))
    cust_input = st.text_area("Tapez votre question")
    
    if st.button("Exécuter la prédiction de tags"):
        if len(cust_input) !=0:
        
            # preparer le texte
            text_wo_html = clean_html(cust_input)
            cleaned_text = text_cleaning(text_wo_html)
            tokenized_text = tokenize(cleaned_text)
            filtered_noun_text = filtering_nouns(tokenized_text)
            lemmatized_text = lemmatize(filtered_noun_text)
            lemmatized_text = ' '.join(lemmatized_text)
            tag_full = predict_tags(lemmatized_text)
            tag_full = list({tag for tag_list in tag_full for tag in tag_list if (len(tag_list) != 0)})
            
            # afficher les résultats
            if len(tag_full) != 0:
                st.markdown("#### Tags prédits")
                for elt in tag_full:
                    st.markdown("<mark style='background-color: lightgray'>**" + str(elt) + "**</mark>",
                                    unsafe_allow_html=True)
            else:
                st.markdown("#### Aucun tag prédit")
              
        else:
            st.info("Tapez votre question")
    
    
if __name__ == '__main__':
        main()
