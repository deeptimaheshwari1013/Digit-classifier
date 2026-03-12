import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import base64
import io
import tensorflow as tf
from scipy import ndimage

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "digit_model.keras"))
print("✅ Model loaded successfully!")

def preprocess(img):
    # Convert to grayscale
    img = img.convert("L")

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy
    pixels = np.array(img)

    # Normalize to 0-1
    pixels = pixels / 255.0

    # ---- KEY FIX: Center the digit like MNIST ----
    # Find the bounding box of the drawn digit
    threshold = 0.1
    rows = np.any(pixels > threshold, axis=1)
    cols = np.any(pixels > threshold, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop to just the digit
        cropped = pixels[rmin:rmax+1, cmin:cmax+1]

        # Add padding (20% on each side)
        pad = 4
        padded = np.pad(cropped, pad, mode='constant', constant_values=0)

        # Resize back to 28x28
        padded_img = Image.fromarray((padded * 255).astype(np.uint8))
        padded_img = padded_img.resize((28, 28), Image.LANCZOS)
        pixels = np.array(padded_img) / 255.0

    return pixels.reshape(1, 28, 28)

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST"
        return response

    try:
        body = request.get_json()
        img_data = body["image"]

        # Decode base64 image
        img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))

        # Preprocess
        pixels = preprocess(img)

        # Predict
        probs = model.predict(pixels, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))
        all_probs = [round(float(p), 4) for p in probs]

        response = jsonify({
            "digit": digit,
            "confidence": round(confidence * 100, 1),
            "probabilities": all_probs
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)