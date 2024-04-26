from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import os

# Initialize the Flask application
app = Flask(__name__)

model = load_model('model.h5')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No image selected for uploading')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        preds = model.predict(x)
        pred_class = "Non-Accident" if preds[0] > 0.5 else "Accident"

        return render_template('index.html', prediction_text=f'Predicted as: {pred_class}')
    else:
        return render_template('index.html', prediction_text='Allowed image types are -> png, jpg, jpeg, gif')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


if __name__ == '__main__':
    app.run(debug=True)
