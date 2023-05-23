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

# save the model using joblib
joblib.dump(model, 'diabetes_model.joblib')
