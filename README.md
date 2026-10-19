# Housing ML Capstone — Prediction API

End-to-end ML project: raw data -> cleaning -> training -> serialized
pipelines -> Flask REST API -> tested -> containerized for deployment.

## Run locally
pip install -r requirements.txt
python training/train_pipeline.py
python app/main.py

Visit http://127.0.0.1:5000/

## Endpoints
- GET /health
- GET /model-info
- POST /predict/price
- POST /predict/category
