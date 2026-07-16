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

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Training / Evaluation / Inference

```bash
python -m src.training.train
python -m src.evaluation.evaluate
python -m src.inference.infer
```

## Testing

```bash
python -m unittest discover -s tests
```

## Deployment

- Docker: `docker build -t fake-image-detector .`
- Compose: `docker compose up --build`
- Nginx config: `deployment/nginx.conf`
- Cloud guides:
  - `deployment/aws-ec2.md`
  - `deployment/render.md`
  - `deployment/huggingface-spaces.md`
