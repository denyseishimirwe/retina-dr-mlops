# Retina DR MLOps Pipeline

An end-to-end machine learning pipeline that classifies retina photos for **diabetic retinopathy (DR)**, a diabetes complication that damages the blood vessels of the eye. The project covers data acquisition, preprocessing, training, testing, retraining, database storage, an API, a web UI, Docker, cloud deployment and load testing.

This builds on my earlier Introduction to Machine Learning summative, which used the CDC Diabetes Health Indicators data (tabular). That project showed that accuracy is misleading on imbalanced classes and that the final decision rule matters more than model size. This one applies the same lessons (macro-F1, class weights, threshold tuning) to image data.

## Links

| | |
|---|---|
| **Video demo (YouTube)** |(https://youtu.be/MsamEX1N4vU) |
| **Live app (AWS)** | (http://13.62.147.146:8501/) |
| **Live API docs (Swagger)** |(http://13.62.147.146:8080/docs) |
| **GitHub repo** | (https://github.com/denyseishimirwe/retina-dr-mlops) |

The video shows (with camera on): a prediction on a retina image, uploading new labelled images, and triggering retraining. It was recorded before the SQLite database integration. The current deployed version includes database-backed upload tracking and retraining.

## Dataset

- **APTOS 2019 retina images**, resized to 224x224 (Kaggle: `sovitrath/diabetic-retinopathy-224x224-2019-data`), 3,662 images in total.
- The original 5 severity grades are grouped into 3 classes:

| Class | Original grades | Images | Share |
|---|---|---:|---:|
| No_DR | No_DR | 1,805 | 49.3% |
| Mild_Moderate | Mild, Moderate | 1,369 | 37.4% |
| Severe_Proliferative | Severe, Proliferate_DR | 488 | 13.3% |

- Split 70 / 15 / 15 (train / validation / test) with `stratify`, so every split keeps the same class mix. The test set has 550 images, 73 of them Severe_Proliferative.

## Data insights (what the visualizations show)

The Visualizations page of the app (data from the `/insights` endpoint) shows three features of the dataset. Each one affects how the model behaves.

1. **Class counts: the data is imbalanced.** No_DR is about half of the images (49.3%), Mild_Moderate is 37.4%, and Severe_Proliferative is only 13.3%. A model can reach high accuracy by favouring the big classes and still miss the rare, most serious cases. This is why I use class weights and report macro-F1 and per-class recall, not accuracy alone. It also explains why Severe is the weakest class in the results below.
2. **Average brightness per class: almost the same across classes.** The three bars on the chart are close in length, so brightness does not separate the classes. It reflects the camera and lighting, not the disease. This means brightness is not a shortcut the model can rely on to tell the classes apart. It also warns that photos taken with different lighting or a different camera could still confuse the model, because the model was not given a brightness signal tied to disease.
3. **Mean red channel per class: heavy overlap between classes.** Retina photos are mostly red and orange, so this feature reflects the overall colour of the image. The averages for the three classes overlap heavily, so simple colour statistics tell us little and colour alone cannot separate severe cases from healthy ones. The model has to learn finer patterns, such as red dots (small bleeds) and yellow patches (deposits), which is why a deep model is needed.

Overall story: the classes are unequal in size, and simple image statistics (brightness and colour) barely separate them, so the model has to learn detailed visual patterns, and the rare Severe class needs special handling.

## Model

- **Transfer learning** with MobileNetV2 (ImageNet weights), frozen at first, then the top 30 layers fine-tuned at a low learning rate.
- Optimization techniques: dropout (0.3), L2 regularization, Adam optimizer, early stopping on validation loss, class weights for the rare Severe class, and threshold tuning chosen on the validation set.
- Augmentation (flip, rotation, brightness) is applied to training images only.
- Saved model: `models/retina_model.keras` and an `.h5` copy `models/retina_model.h5`, with class names and settings in `models/meta.json`.

### Test results (550 unseen images)

| Model | Accuracy | Macro-F1 | Severe recall |
|---|---:|---:|---:|
| Frozen base (before fine-tuning) | 0.873 | 0.806 | 0.603 |
| **Fine-tuned (saved and deployed)** | 0.869 | 0.777 | 0.425 |

Per-class results for the deployed model:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Mild_Moderate | 0.786 | 0.893 | 0.836 |
| No_DR | 0.970 | 0.970 | 0.970 |
| Severe_Proliferative | 0.689 | 0.425 | 0.526 |

Confusion matrix (rows are the true class):

| True \ Predicted | Mild_Moderate | No_DR | Severe_Proliferative |
|---|---:|---:|---:|
| Mild_Moderate | 184 | 8 | 14 |
| No_DR | 8 | 263 | 0 |
| Severe_Proliferative | 42 | 0 | 31 |

### Honest reading of the results

- Fine-tuning lowered validation loss (about 0.364 to 0.339), so the fine-tuned model was kept. On the test set its macro-F1 was lower, because Severe recall fell from 0.603 to 0.425. I deployed the model chosen on validation data and did not pick the model based on the test set, to avoid tuning to the test set.
- The test set has only 73 Severe images, so this difference is noisy.
- No Severe image was predicted as No_DR. The missed Severe cases were labelled Mild_Moderate, so the model errs toward a neighbouring grade and not toward "healthy".
- Threshold tuning on the validation set chose a Severe weight of 1.0, so it gave no gain here.
- Accuracy alone hides the weak Severe class, which is why macro-F1 is the main metric.

The full workflow, with preprocessing, training, charts and metrics, is in `notebook/retina_dr.ipynb`.

## Features

The Streamlit UI has five pages:

1. **Dashboard**: API up-time, retraining status, pending uploads, and last retrain metrics (accuracy, macro-F1).
2. **Predict**: upload one retina image; see the class, confidence and class probabilities.
3. **Visualizations**: class counts, average brightness and mean red channel (from `/insights`), explained in the Data insights section above.
4. **Evaluation**: the reference model's per-class precision, recall and F1, its confusion matrix, and, after a retrain, a comparison of the reference model with the model currently in production (accuracy and macro-F1).
5. **Retrain**: upload many labelled images in one go, then start retraining. A live status panel updates while it runs.

### Retraining process

The application uses SQLite to track uploaded images and their training status. Image files are also stored on disk.

1. **Upload and store:** Users upload labelled retina images through the Streamlit UI. Each image and its label are saved in the SQLite database (`data/retina.db`), together with its file path and upload status. The image file is also stored on disk.
2. **Preprocess:** Uploaded images go through the same preprocessing used for the training data (resize to 224x224 and normalization).
3. **Retrieve pending data:** The retraining process retrieves images marked as `pending` from the SQLite database.
4. **Retrain:** The existing deployed model is loaded as the pre-trained model and fine-tuned using the new labelled images and selected existing training data. The previous model is backed up, and the new model is saved.
5. **Update database:** After successful retraining, the processed upload records are marked as `trained`, with a training timestamp.
6. **Re-evaluate:** The updated model is evaluated, and the latest metrics are displayed on the Dashboard and Evaluation page.
7. **Merge data:** Used uploaded images are moved into the training data folders.

**Retraining trigger:** Retraining can be started using the Retrain button in the UI or `POST /retrain`. It also starts automatically when 50 uploads are waiting (`AUTO_RETRAIN_THRESHOLD`). Pressing Retrain without pending uploads returns "No uploaded data to retrain on".

## Project structure

```text
retina-dr-mlops/
├── README.md
├── app.py                  # Streamlit UI
├── locustfile.py           # load test
├── sample.png              # sample image used by the load test
├── nginx.conf              # load balancer for scaled API containers
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── requirements-api.txt
├── requirements-ui.txt
├── config.toml             # Streamlit theme
├── notebook/
│   └── retina_dr.ipynb
├── src/
│   ├── preprocessing.py    # loading, resizing, datasets
│   ├── model.py            # build, train, retrain, evaluate
│   ├── prediction.py       # load model, predict one image
│   ├── api.py              # FastAPI endpoints
│   └── database.py         # SQLite database operations
├── data/
│   ├── train/              # one folder per class
│   ├── test/
│   ├── uploads/            # uploaded images
│   └── retina.db           # SQLite database (created at runtime)
├── models/
│   ├── retina_model.keras
│   ├── retina_model.h5
│   └── meta.json
└── results/                # Locust CSVs (1, 2 and 3 API containers)
```

The SQLite database and its temporary journal files are excluded from Git using `.gitignore`.

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
git clone https://github.com/denyseishmirwe/retina-dr-mlops.git
cd retina-dr-mlops
docker compose up -d --build
```

The first build takes several minutes because TensorFlow is large. Wait about a minute after it starts for the model to load.

- UI: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8080/docs](http://localhost:8080/docs)

Stop everything with:

```bash
docker compose down
```

The API image uses `tensorflow==2.21.0` and runs on CPU when no GPU is available.

## Load test (Locust)

`sample.png` (one retina image) is already in the repo next to `locustfile.py`. Run:

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
|---|---:|---:|---:|---:|---:|---:|
| 1 | 663 | 2 | 2660 | 1900 | 4600 | 11.40 |
| 2 | 1097 | 1 | 1310 | 1200 | 2700 | 18.55 |
| 3 | 1540 | 1 | 768 | 570 | 1700 | 26.03 |

Tests ran locally with Docker on a MacBook, sending images to `/predict` through an nginx load balancer. Going from 1 to 3 containers raised throughput from 11.4 to 26.0 requests per second (about 2.3x), cut the average response time from 2,660 ms to 768 ms, and cut the 95th percentile from 4,600 ms to 1,700 ms. With one container, requests queued behind each other and response times kept climbing during the test. The gain was not perfectly linear because all containers and Locust shared one laptop's CPU. The few failures were all 502 Bad Gateway errors from nginx under heavy load. Raw Locust CSVs are in the `results/` folder.

## Cloud deployment

Deployed on **AWS EC2** in **eu-north-1** (Stockholm): Ubuntu, Docker Compose, Elastic IP `13.62.147.146`. Security group opens **8501** (UI) and **8080** (API).

Three kinds of containers run with Docker Compose:

- **api**: FastAPI model API (`uvicorn src.api:app`), built from `Dockerfile.api`, running TensorFlow on CPU.
- **nginx**: reverse proxy and load balancer in front of the API, exposed on port 8080. It spreads requests across API containers when the API is scaled.
- **ui**: Streamlit web app (`app.py`), built from `Dockerfile.ui`, exposed on port 8501. It calls the API through nginx.

The `models/` and `data/` folders are mounted as Docker volumes, so retrained models, uploaded images and the SQLite database can persist across container restarts. The live demo runs **one** API container.

### Evaluation in production

An earlier live retraining test used 10 new images and re-evaluated the updated model on the 550 held-out test images:

- Accuracy: 0.865
- Macro-F1: 0.765

The original deployed model had accuracy 0.869 and macro-F1 0.777.

A later AWS database-integration test used one new labelled image. The live application reported:

- Accuracy: 87.5%
- Macro-F1: 0.797
- New images used: 1
- Pending uploads after retraining: 0

The database record was also verified as `trained` with a training timestamp. These are results from separate retraining runs and should not be treated as a direct model comparison.

The live prediction interface was also tested after deployment and returned a No_DR prediction with 99.7% confidence for a known No_DR sample.

The Dashboard (`/uptime`, `/status`, last retrain metrics) and the Evaluation page are the production evaluation view: they show the API is up and report accuracy and macro-F1 after retraining next to the reference model.

## Limitations

- Photos from a different camera or with different framing than the training images can confuse the model.
- The Severe class is rare and the weakest, so this is a learning project and not a medical tool.
- Retrained models are saved to a Docker volume, so they persist. With several API containers, each keeps its own retraining status, so the live demo uses a single container; multi-container retraining would need shared storage.