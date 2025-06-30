from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')  # Make sure model.h5 is in the same folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define your class labels
class_labels = ['arecaceae','anadenanthera','arrabidaea','cecropia','chromolaena',
                'combretum','croton','dipteryx','eucalipto','faramea','hyptis',
                'mabea','matayba','mimosa','myrcia','protium','qualea','schinus',
                'senegalia','serjania','syagrus','tridax','urochloa']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def predict_page():
    return render_template('prediction.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        f = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(150 , 150))  # or (150,150) if not using transfer learning
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalization

        pred = model.predict(x)
        class_index = np.argmax(pred)
        class_name = class_labels[class_index]

        return render_template('prediction.html', pred=class_name)
if __name__ == '__main__':
    app.run(debug=True)
