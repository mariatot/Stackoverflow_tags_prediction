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
