from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

rf_model = pickle.load(open('artifacts/rf_model.pkl', 'rb'))
rf_model = pickle.load(open('artifacts/rf_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = float(request.form['Age'])
        Fare = float(request.form['Fare'])
        Embarked = int(request.form['Embarked'])
        Family = int(request.form['Family'])

        # Prepare features (no scaling needed)
        features = np.array([[Pclass, Sex, Age, Fare, Embarked, Family]])

        # Make prediction
        pred = rf_model.predict(features)
        prediction = "Survived" if pred[0] == 1 else "Did not survive"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)