# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:18:23 2021

@author: Shubhodeep
"""

from flask import Flask,request, render_template
from preprocess import preprocess_text

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
    if(request.method == 'POST'):
        data = request.form['text']
        result = preprocess_text(data)
        my_prediction = result[0]

    return render_template('results.html', prediction = my_prediction)
    

if __name__ == '__main__':
    app.run(debug=True)