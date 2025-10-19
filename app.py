
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model                 # flask api or tensor ke imports
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL = "crop_model_vgg16.h5"                             # trained model or class file ko load krta h
Class = "class_names.txt"

model = load_model(MODEL)
print("[INFO] Loaded trained model.")

with open(Class) as f:                                    # class name ko same order me load krta h jisme train kia tha
    CLASSES = [line.strip() for line in f.readlines()]
print("[INFO] Loaded class names:", CLASSES)


REMEDIES = {                                                   # bimaaariyaa or inki remedy dict me
    "healthy": "No action required. Monitor the crop regularly and maintain proper care.",
    "tomato_blight": "Remove infected plants. Apply fungicides like Mancozeb. Avoid overhead irrigation.\n tubuconzol can be used ",
    "potato_blight": "Apply fungicides and remove infected foliage. Ensure good air circulation. \n Mefenoxam +Mancozeb can be used ",
    "corn_blight": "Use resistant varieties, apply appropriate fungicides, and remove crop debris. \n Trifoxystrobin +Tabuconazol can be used ",
    "apple_blight": "Remove and destroy infected leaves and fruits. Apply copper-based fungicides during early spring. Ensure good air circulation and prune trees properly.",
    "wheat_blight": "Remove infected parts and apply appropriate copper or sulfur-based fungicides. \n tubuconzol or propiconazol orr Tabuconazol with Trifoxystrobin can be used",
    "rust": "Remove infected leaves and apply fungicides like copper, sulfur, or systemic fungicides. \n Chlorathalonil or Mancozeb used to treat this rust",
    "tomato___Tomato_Yellow_Leaf_Curl_Virus": "No direct cure. Isolate infected plants and control whiteflies with insecticides. \n Thiamethoxam & Imidacloprid can be used to treat this",
    "Not_a_Leaf": "Please upload a clear leaf image."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."})

 
    filepath = os.path.join("static", "temp.jpg")                  # upladed image save hoti h temparary
    file.save(filepath)

   
    img = image.load_img(filepath, target_size=(224, 224))         # image process krta h
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # matches training normalization


    preds = model.predict(x)                                       # pridiction ke liye
    idx = np.argmax(preds[0])
    label = CLASSES[idx]
    confidence = float(preds[0][idx])


    print("[DEBUG] Raw prediction:", preds[0])                     # debug krna
    print("[DEBUG] Predicted class:", label, "| Confidence:", confidence)

    remedy = REMEDIES.get(label, "No remedy available.")

    return jsonify({
        "label": label.replace("_", " ").title(),
        "confidence": round(confidence * 100, 2),
        "remedy": remedy
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1221, debug=True)
