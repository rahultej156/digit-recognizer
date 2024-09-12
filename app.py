import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pretrained digit recognition model
model = load_model('model/digit_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        # Process the uploaded image
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))               # Resize to 28x28 pixels
        img = np.array(img)                      # Convert to numpy array
        img = img.reshape(1, 28, 28, 1)          # Reshape for model input
        img = img / 255.0                        # Normalize the pixel values

        # Predict the digit
        prediction = model.predict([img])
        digit = np.argmax(prediction, axis=1)[0]

        return render_template('index.html', prediction=digit)

if __name__ == '__main__':
    app.run(debug=True)
