# Retina DR MLOps Pipeline

An end-to-end machine learning pipeline that classifies retina photos for **diabetic retinopathy (DR)**, a diabetes complication that damages the blood vessels of the eye. The project covers data acquisition, preprocessing, training, testing, retraining, an API, a web UI, Docker, cloud deployment and load testing.

This builds on my earlier Introduction to Machine Learning summative, which used the CDC Diabetes Health Indicators data (tabular). That project showed that accuracy is misleading on imbalanced classes and that the final decision rule matters more than model size. This one applies the same lessons (macro-F1, class weights, threshold tuning) to image data.

## Links

| | |
|---|---|
| **Video demo (YouTube)** | Coming soon |
| **Live app (AWS)** | http://13.49.46.69:8501 |
| **Live API docs** | http://13.49.46.69:8080/docs |
| **GitHub repo** | https://github.com/denyseishimirwe/retina-dr-mlops |

## Dataset

- **APTOS 2019 retina images**, resized to 224x224 (Kaggle: `sovitrath/diabetic-retinopathy-224x224-2019-data`), 3,662 images in total.
- The original 5 severity grades are grouped into 3 classes:

| Class | Original grades | Images |
|---|---|---|
| No_DR | No_DR | 1,805 |
| Mild_Moderate | Mild, Moderate | 1,369 |
| Severe_Proliferative | Severe, Proliferate_DR | 488 |

- Split 70 / 15 / 15 (train / validation / test) with `stratify`, so every split keeps the same class mix. The test set has 550 images, 73 of them Severe_Proliferative.

## Model

- **Transfer learning** with MobileNetV2 (ImageNet weights), frozen at first, then the top 30 layers fine-tuned at a low learning rate.
- Optimization techniques: dropout (0.3), L2 regularization, Adam optimizer, early stopping on validation loss, class weights for the rare Severe class, and threshold tuning chosen on the validation set.
- Augmentation (flip, rotation, brightness) is applied to training images only.
- Saved model: `models/retina_model.keras`, with class names and settings in `models/meta.json`.

### Test results (550 unseen images)

| Model | Accuracy | Macro-F1 | Severe recall |
|---|---|---|---|
| Frozen base (before fine-tuning) | 0.873 | 0.806 | 0.603 |
| **Fine-tuned (saved and deployed)** | 0.869 | 0.777 | 0.425 |

Per-class results for the deployed model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Mild_Moderate | 0.786 | 0.893 | 0.836 |
| No_DR | 0.970 | 0.970 | 0.970 |
| Severe_Proliferative | 0.689 | 0.425 | 0.526 |

Confusion matrix (rows are the true class):

| True \ Predicted | Mild_Moderate | No_DR | Severe_Proliferative |
|---|---|---|---|
| Mild_Moderate | 184 | 8 | 14 |
| No_DR | 8 | 263 | 0 |
| Severe_Proliferative | 42 | 0 | 31 |

**Honest reading of the results**
- Fine-tuning lowered validation loss (about 0.364 to 0.339), so the fine-tuned model was kept. On the test set its macro-F1 was lower, because Severe recall fell from 0.603 to 0.425.
- The test set has only 73 Severe images, so this difference is noisy.
- No Severe image was predicted as No_DR. The missed Severe cases were labelled Mild_Moderate.
- Threshold tuning on the validation set chose a Severe weight of 1.0, so it gave no gain here.
- Accuracy alone hides the weak Severe class, which is why macro-F1 is the main metric.

The full workflow, with charts and metrics, is in `notebook/retina_dr.ipynb`.

## Features

The Streamlit UI has four pages:

1. **Dashboard** — API up-time, retraining status, pending uploads, and last retrain metrics (accuracy, macro-F1).
2. **Predict** — upload one retina image; see the class, confidence and class probabilities.
3. **Visualizations** — class counts, average brightness and mean red channel (from `/insights`).
4. **Retrain** — upload labelled images to `data/uploads/`, then start retraining. The existing model is loaded as the pre-trained weights and fine-tuned. The previous model is backed up, the new one is saved and re-evaluated on the held-out test set, and used images move into the training data. Retraining also starts automatically once 50 uploads are waiting (`AUTO_RETRAIN_THRESHOLD`).

## Project structure

```
retina-dr-mlops/
├── README.md
├── app.py                  # Streamlit UI
├── locustfile.py           # load test
├── nginx.conf              # load balancer for scaled API containers
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements-api.txt
├── requirements-ui.txt
├── config.toml             # Streamlit theme
├── notebook/retina_dr.ipynb
├── src/
│   ├── preprocessing.py    # loading, resizing, datasets
│   ├── model.py            # build, train, retrain, evaluate
│   ├── prediction.py       # load model, predict one image
│   └── api.py              # FastAPI endpoints
├── data/
│   ├── train/              # one folder per class
│   └── test/
├── models/
│   ├── retina_model.keras
│   └── meta.json
└── results/                # Locust CSVs (1, 2 and 3 API containers)
```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Predict one image |
| `POST /upload` | Upload many images with a class label |
| `POST /retrain` | Start retraining |
| `GET /status` | Retraining status and pending uploads |
| `GET /uptime` | API up-time |
| `GET /insights` | Data for the visualizations |
| `GET /health` | Health check |

## Setup

You need Docker and Docker Compose.

```bash
git clone https://github.com/denyseishimirwe/retina-dr-mlops.git
cd retina-dr-mlops
docker compose up -d --build
```

The first build takes several minutes because TensorFlow is large. Wait about a minute after it starts for the model to load.

- UI: http://localhost:8501
- API docs: http://localhost:8080/docs

Stop everything with `docker compose down`.

The API image uses `tensorflow-cpu` so it runs on a CPU EC2 instance.

## Load test (Locust)

Put one retina image next to `locustfile.py` and name it `sample.png`, then run:

```bash
pip install locust
docker compose up -d --build --scale api=1
locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8080 --csv results_1

docker compose up -d --scale api=2
locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8080 --csv results_2

docker compose up -d --scale api=3
locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8080 --csv results_3
```

### Flood request results

50 simulated users for 60 seconds against `/predict`.

| API containers | Requests | Failures | Average (ms) | Median (ms) | 95th percentile (ms) | Requests/sec |
|---|---|---|---|---|---|---|
| 1 | 663 | 2 | 2660 | 1900 | 4600 | 11.40 |
| 2 | 1097 | 1 | 1310 | 1200 | 2700 | 18.55 |
| 3 | 1540 | 1 | 768 | 570 | 1700 | 26.03 |

Tests ran locally with Docker on a MacBook, with 50 simulated users for 60 seconds sending images to `/predict` through an nginx load balancer. Going from 1 to 3 containers raised throughput from 11.4 to 26.0 requests per second (about 2.3x), cut the average response time from 2,660 ms to 768 ms, and cut the 95th percentile from 4,600 ms to 1,700 ms. With one container, requests queued behind each other and response times kept climbing during the test. The gain was not perfectly linear because all containers and Locust shared one laptop's CPU. The few failures were all 502 Bad Gateway errors from nginx under heavy load. Raw Locust CSVs are in the `results/` folder (`sample.png` is already in the repo).

## Cloud deployment

Deployed on **AWS EC2** in **eu-north-1** (Stockholm): Ubuntu, Docker Compose, Elastic IP `13.49.46.69`. Security group opens **8501** (UI) and **8080** (API). The API runs `tensorflow-cpu`. The `models/` and `data/` folders are mounted as Docker volumes, so retrained models and uploaded images survive restarts. The live demo runs **one** API container.

### Evaluation in production

On the live app, uploading 10 new images and pressing Retrain re-evaluated the updated model on the 550 held-out test images: accuracy 0.865 and macro-F1 0.765, compared with 0.869 and 0.777 for the original deployed model. The small change is expected when fine-tuning on only 10 images. The Dashboard (`/uptime`, `/status`, last retrain metrics) is the production evaluation view: it shows the API is up and it reports accuracy and macro-F1 after retraining.

## Limitations

- Photos from a different camera or with different framing than the training images can confuse the model.
- The Severe class is rare and the weakest, so this is a learning project and not a medical tool.
- Retrained models are saved to a Docker volume, so they persist. With several API containers, each keeps its own retraining status, so the live demo uses a single container; multi-container retraining would need shared storage.
