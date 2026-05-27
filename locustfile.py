import random
from locust import HttpUser, task, between

class MLServiceLoadTester(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def send_prediction_request(self):
        model_type = random.choice(["lightgbm", "xgboost", "catboost"])
        gender = random.choice(["male", "female"])
        url_params = f"/predict?model_type={model_type}&gender={gender}"
        
        with self.client.post(url_params, json={}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 500:
                response.failure("Simulated Internal Model Error")