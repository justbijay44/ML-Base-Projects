from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

rf_model = pickle.load(open('artifacts/rf_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inp_classes  = {}

    if request.method == 'POST':
        try:
            inp_classes = {
                'Pclass': int(request.form.get('Pclass', 3)),
                'Sex': int(request.form.get('Sex', 0)),
                'Age': float(request.form.get('Age', 30)),
                'Embarked': int(request.form.get('Embarked', 0)),
                'FamilyGroup': int(request.form.get('Family', 0))
            }

            features = np.array([[v for v in inp_classes.values()]])

            pred = rf_model.predict(features)
            prediction = "✅ Survived" if pred == 1 else "❌ Did not survive"

        except Exception as e:
            prediction = f"Error : {str(e)}"
    return render_template('index.html', prediction=prediction, input_data = inp_classes)

if __name__ == '__main__':
    app.run(debug=True)