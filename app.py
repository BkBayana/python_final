import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Загружаем данные
df = pd.read_csv("heart.csv")

# Разделение на признаки и целевую переменную
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Кодирование категориальных признаков
X = pd.get_dummies(X, drop_first=True)

# Сохраняем названия столбцов после кодирования
model_columns = X.columns

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Flask-приложение
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Получение данных из формы
    features = [float(x) for x in request.form.values()]
    input_df = pd.DataFrame([features], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ])

    # Кодирование категориальных признаков
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Добавляем отсутствующие столбцы и заполняем нулями
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Упорядочиваем столбцы как при обучении
    input_df = input_df[model_columns]

    # Масштабирование
    input_scaled = scaler.transform(input_df)

    # Предсказание
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return render_template("result.html",
                           prediction="Есть риск заболевания" if prediction else "Нет риска заболевания",
                           probability=round(probability * 100, 2))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
