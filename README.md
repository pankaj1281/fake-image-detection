# Fake Image Detection System

Production-ready AI-powered fake image detection platform using **PyTorch**, **FastAPI**, **React**, **Docker**, and cloud deployment guides.

## Features

- Detects: AI-generated, deepfake, GAN-generated, diffusion-generated, manipulated, and real images.
- Ensemble architecture with:
  - EfficientNet-B4
  - ConvNeXt
  - Vision Transformer (ViT)
- Dataset compatibility: CIFAKE, FaceForensics++, DFDC, and custom datasets.
- Training pipeline includes transfer learning, mixed precision, early stopping, checkpointing, TensorBoard, and MLflow.
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
- Explainability: Grad-CAM, attention map, and heatmap output.
- FastAPI endpoints:
  - `POST /predict`
  - `POST /batch_predict`
  - `GET /health`
  - `GET /model_info`
- React frontend includes drag-and-drop upload, preview, prediction confidence, history, and responsive layout.
- Deployment support via Docker, Docker Compose, Nginx, AWS EC2, Render, and Hugging Face Spaces docs.

## Project Structure

```text
fake-image-detector/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── api/
│   └── utils/
├── frontend/
├── deployment/
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Step-by-Step Setup and Run Guide

### 1) Backend environment

Run each command in order from the repository root:

```bash
python -m venv .venv
```
Creates a local virtual environment in `.venv`.

```bash
source .venv/bin/activate
```
Activates the virtual environment (Linux/macOS).

```bash
pip install -r requirements.txt
```
Installs backend dependencies (PyTorch, FastAPI, MLflow, TensorBoard, etc.).

### 2) Prepare training data

Training expects this exact folder layout:

```text
data/
├── train/
│   ├── ai_generated/
│   ├── deepfake/
│   ├── gan_generated/
│   ├── diffusion_generated/
│   ├── manipulated/
│   └── real/
└── val/
    ├── ai_generated/
    ├── deepfake/
    ├── gan_generated/
    ├── diffusion_generated/
    ├── manipulated/
    └── real/
```

If `data/train` or `data/val` is missing, training will stop with a setup message.

### 3) Train the model

```bash
python -m src.training.train
```

What this run does:
- Loads images from `data/train` and `data/val`
- Trains the ensemble model
- Saves checkpoints in `models/checkpoints/`
- Logs metrics to TensorBoard and MLflow
- Uses early stopping automatically

Optional monitoring in separate terminals:

```bash
tensorboard --logdir runs
```

```bash
mlflow ui
```

### 4) Evaluate and inference

```bash
python -m src.evaluation.evaluate
```
Runs example evaluation metrics and prints a report.

```bash
python -m src.inference.infer
```
Runs inference on `data/sample.jpg` (if present).

### 5) Run the backend API

```bash
uvicorn src.api.main:app --reload
```
Starts FastAPI on `http://127.0.0.1:8000`.

Useful endpoints:
- `GET /health`
- `GET /model_info`
- `POST /predict`
- `POST /batch_predict`

### 6) Run the frontend

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal (usually `http://localhost:5173`).

## Testing

```bash
python -m unittest discover -s tests
```

Frontend production build check:

```bash
cd frontend
npm run build
```

## Deployment

- Docker: `docker build -t fake-image-detector .`
- Compose: `docker compose up --build`
- Nginx config: `deployment/nginx.conf`
- Cloud guides:
  - `deployment/aws-ec2.md`
  - `deployment/render.md`
  - `deployment/huggingface-spaces.md`
