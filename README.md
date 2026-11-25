# Retina DR MLOps Pipeline

An end-to-end machine learning pipeline that classifies retina photos for **diabetic retinopathy (DR)**, a diabetes complication that damages the blood vessels of the eye. The project covers data acquisition, preprocessing, training, testing, retraining, an API, a web UI, Docker, cloud deployment and load testing.

This builds on my earlier Introduction to Machine Learning summative, which used the CDC Diabetes Health Indicators data (tabular). That project showed that accuracy is misleading on imbalanced classes and that the final decision rule matters more than model size. This one applies the same lessons (macro-F1, class weights, threshold tuning) to image data.

## Links

| | |
|---|---|
| **Video demo (YouTube)** | `PASTE YOUTUBE LINK HERE` |
| **Live app (AWS)** | `PASTE LIVE UI URL HERE` (UI on port 8501) |
| **Live API docs** | `PASTE LIVE API URL HERE`/docs |
| **GitHub repo** | `PASTE REPO LINK HERE` |

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

1. **Predict** one retina image and see the class, confidence and probabilities.
2. **Visualizations** of the dataset with interpretations (class distribution, brightness, mean red channel).
3. **Upload** many new images for a chosen class (saved to `data/uploads/`).
4. **Retrain** with one button. The uploaded images are preprocessed and the existing model is loaded as the pre-trained model and fine-tuned. The previous model is backed up, the new one is saved and re-evaluated on the test set, and the used images move into the training data.
5. **Auto-trigger:** retraining also starts automatically once 50 uploaded images are waiting (`AUTO_RETRAIN_THRESHOLD`).
6. **Dashboard** with API up-time, retraining status and last retrain metrics.

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
├── notebook/retina_dr.ipynb
├── src/
│   ├── preprocessing.py    # loading, resizing, datasets
│   ├── model.py            # build, train, retrain, evaluate
│   ├── prediction.py       # load model, predict one image
│   └── api.py              # FastAPI endpoints
├── data/
│   ├── train/              # one folder per class
│   └── test/
└── models/
    ├── retina_model.keras
    └── meta.json
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
git clone PASTE_REPO_LINK_HERE
cd retina-dr-mlops
docker compose up -d --build
```

The first build takes several minutes because TensorFlow is large. Wait about a minute after it starts for the model to load.

- UI: http://localhost:8501
- API docs: http://localhost:8080/docs

Stop everything with `docker compose down`.

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
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |

`WRITE A SHORT SUMMARY OF WHAT THE TABLE SHOWS HERE`

## Cloud deployment

Deployed on AWS EC2 with Docker Compose. `PASTE A SHORT DESCRIPTION HERE (instance type, region, ports)`

## Limitations

- Photos from a different camera or with different framing than the training images can confuse the model.
- The Severe class is rare and the weakest, so this is a learning project and not a medical tool.
- Retrained models are stored inside the running container, so rebuilding the container returns to the original model file.