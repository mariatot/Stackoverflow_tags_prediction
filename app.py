import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle, re, html5lib
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub
import pandas as pd
