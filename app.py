import pickle
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, static_folder='static')
def predict_diabetes(age, bmi, blood_pressure, glucose):
    # Convert user inputs to a numpy array
    patient_data = np.array([[age, bmi, blood_pressure, glucose]])

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    # load the diabetes dataset
    df = pd.read_csv('diabetes.csv')

    # select only the four features
    X = df[['Age', 'BMI', 'BloodPressure', 'Glucose']]
    y = df['Outcome']

    # create a pipeline with a standard scaler and logistic regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression())
    ])

    # fit the model to the data
    model.fit(X, y)

    # Make prediction using the trained model
    prediction = model.predict(patient_data)
    print(prediction)
    return prediction[0]

@app.route('/')
def index():
     return render_template('home.html')
    


@app.route('/predict_malaria', methods=['GET','POST'])
def predict_malaria_endpoint():
    if request.method == 'POST':
        # Get the patient data from the POST request
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_pressure = int(request.form['blood_pressure'])
        glucose = int(request.form['glucose'])

        # Make prediction using the predict_diabetes function
        prediction = predict_diabetes(age, bmi, blood_pressure, glucose)

        # Render the template that displays the prediction
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('dibeties.html')
  
@app.route('/predict_diabetes', methods=['GET','POST'])
def predict_diabetes_endpoint():
    if request.method == 'POST':
        # Get the patient data from the POST request
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_pressure = int(request.form['blood_pressure'])
        glucose = int(request.form['glucose'])

        # Make prediction using the predict_diabetes function
        prediction = predict_diabetes(age, bmi, blood_pressure, glucose)

        # Render the template that displays the prediction
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('dibeties.html')
    
@app.route('/predict_dengue', methods=['GET','POST'])
def predict_dengue_endpoint():
    if request.method == 'POST':
        # Get the patient data from the POST request
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_pressure = int(request.form['blood_pressure'])
        glucose = int(request.form['glucose'])

        # Make prediction using the predict_diabetes function
        prediction = predict_diabetes(age, bmi, blood_pressure, glucose)

        # Render the template that displays the prediction
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('dibeties.html')
    
@app.route('/predict_tv', methods=['GET','POST'])
def predict_tv_endpoint():
    if request.method == 'POST':
        # Get the patient data from the POST request
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        blood_pressure = int(request.form['blood_pressure'])
        glucose = int(request.form['glucose'])

        # Make prediction using the predict_diabetes function
        prediction = predict_diabetes(age, bmi, blood_pressure, glucose)

        # Render the template that displays the prediction
        return render_template('prediction.html', prediction=prediction)
    else:
        return render_template('dibeties.html')  
    
app.run(debug=True)
    