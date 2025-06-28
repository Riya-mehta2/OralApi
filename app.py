from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="modelss/model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image like your dataset code (240x240 RGB)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((240, 240))  # Resize to match training size
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'class': 'No image uploaded', 'confidence': 'N/A'})

    image_file = request.files['image']
    image_bytes = image_file.read()

    input_data = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming binary classification
    confidence = float(output_data[0][0])
    predicted_class = 1 if confidence >= 0.5 else 0

    result = {
        "class": "Oral Cancer" if predicted_class == 1 else "No Oral Cancer",
        "confidence": f"{confidence:.2f}"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

