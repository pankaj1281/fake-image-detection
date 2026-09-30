# Fake Image Detection System

Production-ready AI-powered fake image detection platform using **PyTorch**, **FastAPI**, **React**, **Docker**, and cloud deployment guides.

## Features

- Detects fake vs real images by default (with optional 6-class multiclass mode).
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

By default this project uses a **binary setup** (`fake`, `real`) so data collection is easier and training is faster.

Training expects this default folder layout:

```text
data/
├── train/
│   ├── fake/
│   └── real/
└── val/
    ├── fake/
    └── real/
```

If `data/train` or `data/val` is missing, training will stop with a setup message.

Create folders automatically (optional, safe to re-run):

```bash
python -m src.dataset.prepare_structure
```

Prepare train/validation data + manifest automatically:

```bash
python -m src.dataset.prepare_dataset --source-dir data/raw --val-ratio 0.2 --reset-splits
```

Expected source structure (binary default):

```text
data/raw/
├── fake/
└── real/
```

You can paste your own images directly into these folders:
- Put fake/AI-generated images in `data/raw/fake/`
- Put authentic images in `data/raw/real/`

In binary mode, `prepare_dataset` also accepts fake subtype folders (for example `ai_generated/`, `deepfake/`, `gan_generated/`, `diffusion_generated/`, `manipulated/`) and merges them into `fake/` automatically.

This command validates class folders, creates balanced `train/val` splits, and generates `data/dataset_manifest.csv`.

#### Recommended data sources (easy to collect)

- **Fake images**: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge/data), [FaceForensics++](https://github.com/ondyari/FaceForensics)
- **Real images**: [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset), [Open Images](https://storage.googleapis.com/openimages/web/index.html), [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### How many images are needed (binary fake/real)

| Target quality | Train fake | Train real | Val fake | Val real | Total |
|---|---:|---:|---:|---:|---:|
| Quick baseline | 1,500 | 1,500 | 300 | 300 | 3,600 |
| Good local result | 2,500 | 2,500 | 500 | 500 | 6,000 |
| Stronger result | 5,000+ | 5,000+ | 1,000+ | 1,000+ | 12,000+ |

Your 4,000-image dataset is enough to start (target around 2,000 fake + 2,000 real). Focus on clean labels and balanced classes for better accuracy.

#### Optional: switch back to 6-class mode

If you want subtype predictions later, enable multiclass mode before preparing data and training:

```bash
export DATASET_MODE=multiclass
```

Then use class folders:
`ai_generated`, `deepfake`, `gan_generated`, `diffusion_generated`, `manipulated`, `real`.

#### How to upload/copy data into raw folders

After downloading and extracting each dataset, copy files into class folders:

```bash
# binary mode example
cp -r /path/to/extracted/fake_images/* data/raw/fake/
cp -r /path/to/extracted/real_images/* data/raw/real/
```

Then run dataset preparation to create train/val automatically:

```bash
python -m src.dataset.prepare_dataset --source-dir data/raw --val-ratio 0.2 --reset-splits
```

You do **not** need to manually create `data/train` or `data/val` splits from raw data. The command above creates balanced fake/real train and val folders for you.

### 3) Train the model

Fast iteration mode (recommended in VS Code while tuning):

```bash
python -m src.training.train --profile fast
```

Faster run on local machine (good for VS Code iteration):

```bash
python -m src.training.train --profile fast --backbones convnext_tiny --epochs 4 --image-size 160 --batch-size 16
```

Best-accuracy mode (full ensemble):

```bash
python -m src.training.train --profile accurate
```

What this run does:
- Loads images from `data/train` and `data/val`
- Trains the ensemble model
- Saves checkpoints in `models/checkpoints/`
- Logs metrics to TensorBoard and MLflow
- Uses early stopping automatically

Common speed/accuracy overrides:

```bash
# tune data loading and throughput
python -m src.training.train --profile fast --num-workers 8 --batch-size 32

# custom backbone subset
python -m src.training.train --profile accurate --backbones efficientnet_b4,convnext_tiny
```

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

### Fast + accurate training checklist

1. Use `src.dataset.prepare_dataset` to avoid class imbalance and split mistakes.
2. First run `--profile fast` to verify data and pipeline quickly.
3. For final training, run `--profile accurate`.
4. Keep fake/real class counts similar.
5. Increase `--num-workers` and keep data on SSD for faster loading.

### Other places you can train faster

- **Google Colab** (free + easy GPU start)
- **Kaggle Notebooks** (free GPU sessions close to Kaggle datasets)
- **Paperspace Gradient**
- **AWS EC2 GPU** / **GCP Vertex AI** / **Azure ML** (paid, longer and larger runs)

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
