from flask import Flask, render_template, json, jsonify, request, Response
import io
import jsonpickle
import numpy as np
# import cv2
from PIL import Image
from ModelClass import LgModel
from scripts.label_image import predict_result

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 mb image

@app.route("/")
def home():
    return render_template('Interior_design.html')

@app.route("/classify", methods=['POST'])
def get_image():
    fakePredictions = {
        'bohemian': 0.5,
        'scandinavian': 0.5,
        'industrial': 0.5,
        'coastal': 0.5
    }
    
    file_object = request.files['photo']
    photo = file_object.read()
    prediction = predict_result(photo)
    fakePredictions.update({prediction: 0.8 })

    return jsonify(fakePredictions)

# def get_filename():
#     user_data = request.get_json[]
#     n,r=int(user_data['n'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
