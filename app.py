from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image
import keras.utils as image
import keras
from keras.utils import CustomObjectScope
import tensorflow as tf
from tensorflow.keras import backend as K
import base64
from io import BytesIO


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


app = Flask(__name__)
CORS(app)
W = H = 256
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss, "iou" : iou }):
    model = keras.models.load_model("https://drive.google.com/file/d/1fgyIHZwFFynm8xtlOl4dxMN7A1Vwylzt/view?usp=drive_link")

@app.route('/predict', methods=['POST'])
def predict():
    print ("image requested ..")
    image_file = request.files['image']
    
    image = Image.open(image_file.stream).convert("RGB")
    image = image.resize((W, H))
    x = np.array(image) / 255.0
    x = np.expand_dims(x, axis=0)

    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255

    img_mask = x * (1 - y_pred) * 255
    img_mask = np.squeeze(img_mask, axis=0)

    cat_images = np.concatenate([np.array(image), line, y_pred, line, img_mask], axis=1)
    predicted_image = Image.fromarray(cat_images.astype(np.uint8))
    
    # Encode image data to base64
    buffered = BytesIO()
    predicted_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print ("finished ..")


    return jsonify({'outputImageData': encoded_image})

if __name__ == '__main__':
    app.run( port=5000)
