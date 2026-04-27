from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("jellyfish_model.keras")

class_names = [
    'Moon_jellyfish',
    'barrel_jellyfish',
    'blue_jellyfish',
    'compass_jellyfish',
    'lions_mane_jellyfish',
    'mauve_stinger_jellyfish'
]

def preprocess(file_path):
    img =Image.open(file_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file_path = "temp.jpeg"
    file.save(file_path)
    img = preprocess(file_path)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    print(prediction)
    return render_template("result.html", prediction_text=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)


