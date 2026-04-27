from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

model = load_model("jellyfish_model.h5")
class_names =

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


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

    return render_template("result.html", prediction_text=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)


