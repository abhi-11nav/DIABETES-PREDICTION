import numpy as np
import pickle
from flask import Flask, request, render_template


app = Flask(__name__, template_folder='./templates')
app.debug=True
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', prediction_text="Your output is {}".format(prediction))


if __name__ == "__main__":
    app.run(use_reloader=False,debug=True)
    