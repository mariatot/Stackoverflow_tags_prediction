import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle, re, html5lib
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub
import pandas as pd

app = Flask(__name__)
classifier_use = pickle.load(open('model_fin_supervised.pkl','rb'))
mlb = pickle.load(open('mlb_model_fin.pkl','rb'))
stop_words_df = pd.read_csv("stopwords-en.csv", sep=',')
stop_words = stop_words_df.columns.to_list()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def clean_text(text):
    """
        préparation du texte
    """
    text = text.lower() 
    cleaned_text = BeautifulSoup(text, 'html5lib').text
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    cleaned_text = word_tokenize(cleaned_text)
    cleaned_text = [word for word in cleaned_text if word not in stop_words]
    cleaned_text = [WordNetLemmatizer().lemmatize(word) for word in cleaned_text]
    return ' '.join(cleaned_text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input = data['text']
    clean_input = clean_text(input)
    input_feature = embed([clean_input])
    prediction = classifier_use.predict(input_feature)
    readable_prediction = mlb.inverse_transform(prediction)
    return render_template('index.html', prediction_text=readable_prediction)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    input =  data['text']
    clean_input = clean_text(input)
    input_feature = embed(clean_input)
    prediction = classifier_use.predict(input_feature)
    readable_prediction = mlb.inverse_transform(prediction)
    return jsonify(str(readable_prediction))


if __name__ == '__main__':
    app.run(debug=True)

