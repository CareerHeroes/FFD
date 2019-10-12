from flask import request, render_template, redirect, url_for # Import to do all routing stuff
from app import app # Import app to use routes 
import stripe # Import for payment process
import os # Import to get files
from dotenv import load_dotenv # Import to load production.env file
from model_deploy import * # Import to build the model and predict a image
import tensorflow as tf # Import to build the graph for the model
from fastai import *
from fastai.vision import *


model = build_model() # Build the model with the specific json and h5 weights file
graph = tf.get_default_graph() # Get the default graph from tensorflow

# Load Environment Variables File
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, './production.env'))

# Get the pub and secret key from the env file and set the stripe api key
pub_key = os.getenv('pub_key')
secret_key = os.getenv('secret_key')
stripe.api_key = secret_key

# Define the allowed file extensions that can be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Defining the model labels
labels = ['fake', 'real']

export_file_url = 'https://drive.google.com/uc?export=download&id=1-Rlv4jsQa0XGsDNMvadntQhQj5r93sj-'
export_file_name = 'export.pkt'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise



@app.route('/')
def index():
	print("1")
	return render_template('index.html', pub_key=pub_key)


@app.route('/image_upload')
def image_upload():
	return render_template('image_upload.html', predictions=[])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def predict():
	print("2")
	if 'file' not in request.files:
		return render_template('image_upload.html', predictions=[])
	
	file = request.files['file']
	if file.filename == '':
		return render_template('image_upload.html', predictions=[])
	
	if file and allowed_file(file.filename):
		global graph
		with graph.as_default():
			predictions = learn.predict(file)
		return render_template('image_upload.html', predictions=list(predictions[0]))

	return render_template('image_upload.html', predictions=[])


@app.route('/pay', methods=['POST', 'GET'])
def pay():
	return redirect(url_for('image_upload'))

