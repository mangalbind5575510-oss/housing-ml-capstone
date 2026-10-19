import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

from common.preprocessing import HousingPreprocessor  # noqa: F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
METADATA_PATH = os.path.join(ROOT, "output", "metadata.json")

REQUIRED_FIELDS = {
    "Area_sqft": (int, float),
    "Bedrooms": (int, float),
    "Bathrooms": (int, float),
    "Age_years": (int, float),
    "Location": (str,),
    "Furnishing": (str,),
    "Parking": (str,),
}
VALID_LOCATIONS = ["Downtown", "Suburb", "Rural", "Uptown", "Waterfront"]
VALID_FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
VALID_PARKING = ["Yes", "No"]

app = Flask(__name__)

print("Loading serialized pipelines ...")
price_pipeline = joblib.load(os.path.join(MODELS_DIR, "price_pipeline.joblib"))
category_pipeline = joblib.load(os.path.join(MODELS_DIR, "category_pipeline.joblib"))
with open(METADATA_PATH) as f:
    METADATA = json.load(f)
print("Pipelines loaded successfully.")


def validate_payload(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return None, f"Missing required field(s): {', '.join(missing)}"

    cleaned = {}
    for field, types in REQUIRED_FIELDS.items():
        value = data[field]
        if not isinstance(value, types):
            return None, f"Field '{field}' must be of type {types[0].__name__}, got {type(value).__name__}"
        cleaned[field] = value

    if cleaned["Location"] not in VALID_LOCATIONS:
        return None, f"'Location' must be one of {VALID_LOCATIONS}"
    if cleaned["Furnishing"] not in VALID_FURNISHING:
        return None, f"'Furnishing' must be one of {VALID_FURNISHING}"
    if cleaned["Parking"] not in VALID_PARKING:
        return None, f"'Parking' must be one of {VALID_PARKING}"
    for numeric_field in ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years"]:
        if cleaned[numeric_field] < 0:
            return None, f"'{numeric_field}' cannot be negative"

    return cleaned, None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(METADATA), 200


@app.route("/predict/price", methods=["POST"])
def predict_price():
    payload = request.get_json(silent=True)
    cleaned, error = validate_payload(payload)
    if error:
        return jsonify({"error": error}), 400

    X = pd.DataFrame([cleaned])
    prediction = price_pipeline.predict(X)[0]
    return jsonify({
        "predicted_price": round(float(prediction), 2),
        "currency": "USD",
        "model": METADATA["price_model"]["algorithm"],
    }), 200


@app.route("/predict/category", methods=["POST"])
def predict_category():
    payload = request.get_json(silent=True)
    cleaned, error = validate_payload(payload)
    if error:
        return jsonify({"error": error}), 400

    X = pd.DataFrame([cleaned])
    pred_class = int(category_pipeline.predict(X)[0])
    proba_high = float(category_pipeline.predict_proba(X)[0][1])
    return jsonify({
        "category": "High" if pred_class == 1 else "Low",
        "probability_high": round(proba_high, 4),
        "price_threshold_used": METADATA["category_model"]["price_median_threshold"],
        "model": METADATA["category_model"]["algorithm"],
    }), 200


DEMO_FORM_HTML = """
<!doctype html>
<html>
<head>
  <title>Housing ML Capstone - Demo</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; color: #222; }
    h1 { color: #2E5D9F; }
    label { display: block; margin-top: 12px; font-weight: bold; }
    input, select { width: 100%; padding: 6px; margin-top: 4px; box-sizing: border-box; }
    button { margin-top: 20px; padding: 10px 18px; background: #2E5D9F; color: white; border: none; border-radius: 4px; cursor: pointer; }
    #result { margin-top: 20px; padding: 14px; background: #EAF0FA; border-radius: 6px; white-space: pre-wrap; font-family: monospace; }
  </style>
</head>
<body>
  <h1>Housing Price &amp; Category Predictor</h1>
  <p>Demo UI for the Week 4 capstone ML API. Fill in the fields and submit.</p>
  <form id="f">
    <label>Area (sqft)</label><input name="Area_sqft" type="number" value="1800" required>
    <label>Bedrooms</label><input name="Bedrooms" type="number" value="3" required>
    <label>Bathrooms</label><input name="Bathrooms" type="number" value="2" required>
    <label>Age (years)</label><input name="Age_years" type="number" value="10" required>
    <label>Location</label>
    <select name="Location">
      <option>Downtown</option><option>Suburb</option><option>Rural</option>
      <option>Uptown</option><option>Waterfront</option>
    </select>
    <label>Furnishing</label>
    <select name="Furnishing">
      <option>Unfurnished</option><option>Semi-Furnished</option><option>Fully-Furnished</option>
    </select>
    <label>Parking</label>
    <select name="Parking"><option>Yes</option><option>No</option></select>
    <button type="button" onclick="predict('price')">Predict Price</button>
    <button type="button" onclick="predict('category')">Predict Category</button>
  </form>
  <div id="result"></div>
<script>
async function predict(kind) {
  const form = document.getElementById('f');
  const data = {
    Area_sqft: Number(form.Area_sqft.value),
    Bedrooms: Number(form.Bedrooms.value),
    Bathrooms: Number(form.Bathrooms.value),
    Age_years: Number(form.Age_years.value),
    Location: form.Location.value,
    Furnishing: form.Furnishing.value,
    Parking: form.Parking.value,
  };
  const res = await fetch('/predict/' + kind, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
  });
  const json = await res.json();
  document.getElementById('result').textContent = JSON.stringify(json, null, 2);
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(DEMO_FORM_HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
