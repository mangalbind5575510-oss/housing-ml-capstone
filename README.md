# Housing ML Capstone — Prediction API

End-to-end ML project: raw data -> cleaning -> training -> serialized
pipelines -> Flask REST API -> tested -> containerized for deployment.

## Run locally
pip install -r requirements.txt
python training/train_pipeline.py
python app/main.py

Visit https://housing-ml-capstone.onrender.com

## Endpoints
- GET /health
- GET /model-info
- POST /predict/price
- POST /predict/category
