import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

df = pd.read_csv("heart.csv")
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X = pd.get_dummies(X, drop_first=True)
model_columns = X.columns


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

# Create a plot only once
def create_plot():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='HeartDisease', multiple='stack', bins=30)
    plt.title("Age Distribution by Heart Disease")
    plt.xlabel("Age")
    plt.ylabel("Count")
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig("static/plot.png")
    plt.close()

create_plot()

# Home route
@app.route("/")
def home():
    return render_template("form.html", accuracy=round(accuracy * 100, 2))

# Prediction route
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
                           prediction="At risk of heart disease" if prediction else "No risk of heart disease",
                           probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
