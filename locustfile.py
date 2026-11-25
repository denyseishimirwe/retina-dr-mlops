from locust import HttpUser, task, between

class RetinaUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        with open("sample.png", "rb") as f:
            self.client.post("/predict", files={"file": ("sample.png", f, "image/png")})