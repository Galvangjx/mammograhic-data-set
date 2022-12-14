import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
# load the model
bestmodel = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = bestmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = list(int(x) for x in request.form.values())
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = bestmodel.predict(final_input)
    return render_template("home.html", prediction_text=f"The predicted outcome of mammograhphy is {int(output)}.")

if __name__=="__main__":
    app.run(debug=True)
