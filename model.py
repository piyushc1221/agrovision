import os
import io
import numpy as np
from PIL import Image
import argparse

try:
    import tensorflow as tf               #idhar tensorflow import kia h
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
except Exception as e:
    raise ImportError(
        "TensorFlow / Keras not installed. Install with:\n"
        "pip install tensorflow pillow numpy flask\n"
        f"Original error: {e}"
    )

MODEL_PATH = "crop_model_vgg16.h5"  #CNN traine model ka ye path hai
CLASS_LABELS = [
    "Healthy",
    "Tomato_Blight",
    "Potato_Blight",           #bimariiiiiiii
    "Corn_Blight",
    "apple_Blight",
    "wheat_Blight",
    "rust"
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]  # Replace with your classes
REMEDY_MAP = {
    "Healthy": "No action required. Monitor the crop regularly and maintain proper care.",        # bimaariyo ki remedy dict me
    "Tomato_Blight": "Remove infected plants. Apply fungicides like Mancozeb. Avoid overhead irrigation.",
    "Potato_Blight": "Apply fungicides and remove infected foliage. Ensure good air circulation.",
    "Corn_Blight": "Use resistant varieties, apply appropriate fungicides, and remove crop debris.",
    "apple_Blight":"Remove and destroy infected leaves and fruits. \n Apply copper-based or sulfur fungicides during early spring and at leaf emergence. \n Ensure good air circulation — prune trees properly. \nWater at the base of the tree; avoid overhead irrigation.",
    "wheat_Blight":"Remove infected parts and apply appropriate copper or sulfur-based fungicides.",
    "rust":"Remove infected leaves and apply fungicides like copper, sulfur, or systemic fungicides.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":"this do not have a direct cure save other plants from this and use fungiside to cure it"
}

IMG_SIZE = (224, 224)

if not os.path.exists(MODEL_PATH):                           #model ki loading traine wale ki
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(f"[INFO] Loaded trained model from {MODEL_PATH}")

def preprocess_image(image_path):
    """Load image and preprocess for model prediction"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = vgg_preprocess(arr)          #vgg16 ki processing
    return arr


def predict(image_path):                  #pridictionnnnn
    img_arr = preprocess_image(image_path)
    preds = model.predict(img_arr, verbose=0)[0]  # shape (num_classes,)
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = CLASS_LABELS[top_idx]
    remedy = REMEDY_MAP.get(label, "No remedy available. Consult an expert.")
    return {
        "label": label,
        "confidence": confidence,
        "remedy": remedy,
        "raw_scores": preds.tolist()
    }


def make_flask_app():                  # flask api for app route
    from flask import Flask, request, jsonify

    app = Flask("crop_disease_api")

    @app.route("/predict", methods=["POST"])
    def predict_route():
        if "image" not in request.files:
            return jsonify({"error": "No image file. Use key 'image'."}), 400
        file = request.files["image"]
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            arr = np.array(img).astype("float32")
            arr = np.expand_dims(arr, 0)
            arr = vgg_preprocess(arr)
            preds = model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            label = CLASS_LABELS[top_idx]
            confidence = float(preds[top_idx])
            remedy = REMEDY_MAP.get(label, "No remedy available.")
            response = {
                "label": label,
                "confidence": confidence,
                "remedy": remedy,
                "raw_scores": preds.tolist()
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/", methods=["GET"])
    def index():
        return (
            "<h3>Crop Disease Prediction API</h3>"
            "<p>POST an image to <code>/predict</code> as form-data (key='image').</p>"
        )

    return app


def main():                        #commandline
    parser = argparse.ArgumentParser(description="Crop Disease Prediction CLI/API")
    parser.add_argument("--image", type=str, help="Path to a leaf image to predict.")
    parser.add_argument("--serve", action="store_true", help="Start Flask API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            return
        res = predict(args.image)
        print("\n=== Prediction Result ===")
        print(f"Predicted disease: {res['label']}")
        print(f"Confidence score : {res['confidence']:.4f}")
        print(f"Suggested remedy : {res['remedy']}")
        print("========================\n")

    if args.serve:
        print(f"[INFO] Starting Flask API on http://{args.host}:{args.port}")
        app = make_flask_app()
        app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
