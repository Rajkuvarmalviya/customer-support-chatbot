import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from textblob import TextBlob  

# NLTK setup (Online error avoid karne ke liye)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# --- Load Data ---
# Cache resource use karte hain taaki har baar file load na ho
@st.cache_resource
def load_resources():
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
    return intents, words, classes, model

intents, words, classes, model = load_resources()

# --- Helper Functions (Same as before) ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- Streamlit UI Code ---
st.title("🤖 Customer Support Chatbot")
st.write("Welcome! Ask me about Laptops, Mobiles, or Order Status.")

# Chat history ko maintain karna (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purani chat history dikhana
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User ka input lena
if prompt := st.chat_input("Type your message here..."):
    # User ka message display karein
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Bot ka response generate karein
    ints = predict_class(prompt)
    res = get_response(ints, intents)
    blob = TextBlob(prompt)
    sentiment = blob.sentiment.polarity  # -1 (Negative) se +1 (Positive)
    
    mood_emoji = "😐" # Default Neutral
    if sentiment > 0.3:
        mood_emoji = "😊 (Positive)"
    elif sentiment < -0.3:
        mood_emoji = "😡 (Negative)"

    # Bot ka message display karein
    with st.chat_message("assistant"):
        st.markdown(res)
        # Niche mood bhi dikhayein
        st.caption(f"Detected User Sentiment: {mood_emoji}")
        
    st.session_state.messages.append({"role": "assistant", "content": res})

    