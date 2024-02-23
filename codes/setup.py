from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS from flask_cors
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

MODEL = tf.keras.models.load_model("D:\\code of hackathon\\codes\\2")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route("/")
def index():
    return render_template("index.html")

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to have shape (256, 256, 3)
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    additional_message = ""
    if predicted_class == "Late Blight":
        additional_message = ("Late Blight:<br><br>"
                              "Cause: Late blight is caused by the pathogenic oomycete organism, Phytophthora infestans.</br>"
                              "Symptoms: It typically starts with small, water-soaked lesions on leaves that rapidly expand and turn brown. These lesions may also appear on stems and fruits. Under moist conditions, a white, fuzzy growth (fungal sporulation) may be visible on the undersides of leaves.<br>"
                              "Spread: Late blight can spread rapidly, especially in cool, wet weather conditions. It can devastate entire tomato or potato crops if not managed effectively.<br>"
                              "Control: Management strategies include using resistant cultivars, practicing crop rotation, applying fungicides preventatively, and ensuring good air circulation around plants to reduce humidity levels.")
    elif predicted_class == "Early Blight":
        additional_message = ("Early Blight:<br><br>"
                              "Cause: Early blight is caused by the fungus Alternaria solani.<br>"
                              "Symptoms: Symptoms usually appear on the lower leaves first as small, dark brown spots with concentric rings. These spots may enlarge and cause the leaves to yellow and eventually die. Lesions can also appear on stems and fruit.<br>"
                              "Spread: Early blight is favored by warm, humid conditions. It can overwinter on plant debris in the soil and spread through splashing water and wind.<br>"
                              "Control: Cultural practices such as crop rotation, mulching, and pruning can help reduce the spread of early blight. Fungicides can also be used preventatively. Planting resistant tomato varieties can be effective in managing early blight.")
    elif predicted_class == "Healthy":
        additional_message = "Healthy:<br><br>This potato plant appears to be healthy."

    return jsonify({
        'class': predicted_class,
        'confidence': float(confidence),
        'additional_message': additional_message
    })

if __name__ == "__main__":
    app.run(debug=True)
    