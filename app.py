import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Changed
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("heart.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X = pd.get_dummies(X, drop_first=True)

model_columns = X.columns


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = LogisticRegression()
model.fit(X_scaled, y)


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    input_df = pd.DataFrame([features], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ])


    input_df = pd.get_dummies(input_df, drop_first=True)


    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0


    input_df = input_df[model_columns]


    input_scaled = scaler.transform(input_df)


    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return render_template("result.html",
                           prediction="Есть риск заболевания" if prediction else "Нет риска заболевания",
                           probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
