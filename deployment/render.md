# Render Deployment Guide

1. Create a Web Service from this repo.
2. Build command: `pip install -r requirements.txt`.
3. Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.
4. Add environment variables as needed.
