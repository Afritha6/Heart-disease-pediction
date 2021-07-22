from flask import Flask, render_template, request
import pickle
import numpy as np

filename='HeartPatients-prediction-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__, static_folder='static')



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    thal = int(request.form['thal'])
    exang = int(request.form['exang'])
    cp = int(request.form['cp'])
    ca = int(request.form['ca'])
    fbs = int(request.form['fbs'])
    slope = int(request.form['slope'])
    oldpeak = float(request.form['oldpeak'])

    arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, ca, slope, thal,oldpeak]])

    y = classifier.predict(arr)
        
        # No heart disease
    if y == 0:
        return render_template('nodisease.html')

    # y=1,2,4,4 are stages of heart disease
    else:
        return render_template('heartdisease.html', stage=int(y))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
