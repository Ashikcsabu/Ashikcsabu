from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C://Users//ASHIK C SABU//Desktop//placement.csv")
df = pd.DataFrame(data)

# Separating dependent and independent variables
X = df.cgpa.values.reshape(-1, 1)
y = df.package.values.reshape(-1, 1)

# Train the Linear Regression model
lr = LinearRegression()
lr.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])

    # Predict the package for the entered CGPA
    prediction = lr.predict([[cgpa]])

    return render_template('result.html', cgpa=cgpa, predicted_package=prediction[0, 0])

if __name__ == '__main__':
    app.run(debug=True)
