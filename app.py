from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# Load Data
print("Loading Model and Files...")
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = "I am sorry, I do not understand."
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- ROUTES ---

@app.route("/")
def home():
    # Make sure 'index.html' is inside 'templates' folder
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    # --- SPECIAL LOGIC: ORDER TRACKING ---
    # Agar user koi Number type karta hai, to use Order ID maano
    if userText.isdigit():
        possible_status = [
            "Processing in Warehouse 🏭",
            "Shipped via BlueDart 🚚",
            "Out for Delivery 🛵",
            "Delivered Successfully ✅"
        ]
        status = random.choice(possible_status)
        return f"📦 Order #{userText}: <b>{status}</b>"
    
    # --- NORMAL AI LOGIC ---
    ints = predict_class(userText)
    res = get_response(ints, intents)
    return res

if __name__ == "__main__":
    app.run(debug=True, port=5000)