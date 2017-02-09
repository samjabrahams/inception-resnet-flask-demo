import os
import os.path
import random
from flask import Flask, request, g, redirect, render_template, flash, \
    send_from_directory
from werkzeug.utils import secure_filename
import model

################################################
# Flask App configuration
################################################
# Simple, unsafe Flask settings
app = Flask(__name__, instance_relative_config=True)
app.config.from_object(__name__)
app.config.update({
    'DATABASE': os.path.join(app.root_path, 'serving.db'),
    'SECRET_KEY': 'development key',
    'USERNAME': 'admin',
    'PASSWORD': 'default',
    'UPLOAD_FOLDER': 'temp'
})

################################################
# Model initialization
################################################
if os.environ.get("FLASK_DEBUG") == "1":
    # This prevents multiple sessions from being created in DEBUG mode
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        sess = model.Session()
else:
    sess = model.Session()


################################################
# Functions for descriptions of model output
################################################
def create_descriptions():
    """Converts static/descriptions.txt into a list of strings"""
    descriptions = []
    with app.open_resource('static/descriptions.txt', mode='r') as f:
        for line in f:
            descriptions.append(line)
    return descriptions


def get_descriptions():
    """Returns the Inception-ResNet string description list"""
    if not hasattr(g, 'descriptions'):
        g.descriptions = create_descriptions()
    return g.descriptions


################################################
# Routes
################################################
@app.route('/temp/<filename>')
def uploaded_file(filename):
    """Allows us to serve temporarily hosted images in the temp directory"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET'])
def home():
    """Basic index page rendering"""
    return render_template('predict.html')


@app.route('/', methods=['POST'])
def predict():
    """Classifies JPEG image passed in as POST data
    
    Assuming a JPEG file is passed in (as raw bytes), this function saves the 
    image to a the local temp directory, passes in the image to the TensorFlow
    model, and returns the top-5 guesses and path to the saved image to be 
    rendered to the client.

    NOTE: This function is NOT SAFE. Strictly for demonstration purposes. Does
    not do any safe-checking of the data being saved locally. Only use locally.
    """
    results = []
    filename = None
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
        filename = os.path.join(app.config['UPLOAD_FOLDER'], '{}.jpg'.format(
            random.randint(0, 999999999)))
        file.save(filename)
        file.seek(0)
        data = file.read()
        feed_dict = {model.get_input(sess): data}
        prediction = sess.run(model.get_predictions(sess), feed_dict)
        top_k = prediction.argsort()[0][-5:][::-1]
        descriptions = get_descriptions()
        for idx in top_k:
            description = descriptions[idx]
            score = prediction[0][idx]
            print('{} (score = {})'.format(description, score))
            results.append((description, score))
    return render_template('predict.html', results=results, filename=filename)
