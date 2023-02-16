import pickle
# from flask import Flask , request , app , jsonify , 
# from flask import Flask
from flask import Flask ,request,app,jsonify,render_template
import numpy as np
import pandas as pd

# Starting point , from where application strat run.
app = Flask(__name__)
# Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

# Basically a root , from where local host initial starting point start from.
@app.route('/')

# This will return home html page.
def  home():
    return render_template('home.html')

# To create API for request 

@app.route('/predict_api',method=['POST'])

def predict():
    data = request.json['data'] # Basically from this line it explain that when '/predict_api' hit , the request will give output
    print(data)                 # in form of json of data. and stored into variable named as data.    
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.prediction(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)