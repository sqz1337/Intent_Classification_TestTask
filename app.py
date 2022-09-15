import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import scipy
import joblib

app = Flask(__name__)
with open('pipe_sgd.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict():
    message = request.form['message']
    prediction = model.predict_proba([message])

    n = 3

    prediction = model.predict_proba([message])
    top_n_lables_idx = np.argsort(-prediction, axis=1)[:, :n]
    top_n_probs = np.round(-np.sort(-prediction), 3)[:, :n]
    top_n_labels = [model.classes_[i] for i in top_n_lables_idx]

    results = list(zip(top_n_labels, top_n_probs))
    predicted_intent = []
    proba_intent = []
    for i in range(3):
        predicted_intent.append(results[0][0][i])
        proba_intent.append(results[0][1][i])

    return render_template("result.html", prediction = [predicted_intent,proba_intent])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5357, debug=True)
