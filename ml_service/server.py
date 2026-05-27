import time
import random
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Histogram, Counter

# Импортируем инструменты для работы с предобученной моделью Hugging Face
import torch
from transformers import pipeline

app = Flask(__name__)
metrics = PrometheusMetrics(app, group_by='endpoint')

# --- Инициализация предобученной модели на GPU ---
print("Проверка доступности GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Модель будет запущена на устройстве: {device.upper()}")

print("Загрузка предобученной модели DistilBERT (Sentiment Analysis)...")
# Скачиваем официальную модель для анализа тональности текста и переносим её на GPU
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
print("Предобученная модель успешно загружена в память!")

# Метрики для Grafana
INF_TIME = Histogram(
    "model_inference_seconds", 
    "Time spent running model inference", 
    labelnames=["model_type", "gender"]
)
ERRORS_COUNTER = Counter(
    "ml_service_errors_total", 
    "Total number of simulated errors", 
    labelnames=["model_type", "gender"]
)

@app.route("/predict", methods=["POST"])
def predict():
    model_type = request.args.get("model_type", "lightgbm")
    gender = request.args.get("gender", "male")
    
    if random.random() < 0.03:
        ERRORS_COUNTER.labels(model_type=model_type, gender=gender).inc()
        return jsonify({"error": "Internal Model Error"}), 500

    start_inf = time.time()
    
    # --- НАСТОЯЩИЙ ИНФЕРЕНС ПРЕДОБУЧЕННОЙ НЕЙРОСЕТИ ---
    # Генерируем случайный текст отзыва в зависимости от пола, чтобы сохранить логику
# Список реальных отзывов для обработки Бертом
    male_reviews = [
        "Perfect app! Highly recommend to everyone.",
        "It works fine, but the interface is a bit clunky.",
        "Awesome performance on this new GPU cluster!"
    ]

    female_reviews = [
        "I am very disappointed, it crashes all the time.",
        "Good customer support, resolved my issue quickly.",
        "The service is okay, nothing special to be honest."
    ]

# Выбираем случайный текст в зависимости от пола
    if gender == "male":
        text_sample = random.choice(male_reviews)
    else:
        text_sample = random.choice(female_reviews)
        
    # Запускаем нейросеть (она делает честный математический расчет на GPU Tesla T4)
    result = classifier(text_sample)[0]
    
    # Извлекаем вероятность (score от 0.0 до 1.0)
    real_proba = result['score']
    
    if model_type == "catboost":
        time.sleep(random.uniform(0.06, 0.08))
    elif model_type == "xgboost":
        time.sleep(random.uniform(0.03, 0.05))
    else:
        time.sleep(random.uniform(0.005, 0.01)) # Наш дефолтный быстрый вариант
        
    inf_duration = time.time() - start_inf
    INF_TIME.labels(model_type=model_type, gender=gender).observe(inf_duration)

    return jsonify({
        "status": "success",
        "model_used": model_type,
        "gender": gender,
        "gpu_active": torch.cuda.is_available(), # Флаг активности GPU для проверки
        "sentiment_label": result['label'],      # Что ответила нейросеть (POSITIVE/NEGATIVE)
        "probability": round(float(real_proba), 4) # Честная уверенность модели в своем ответе
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
