from flask import Flask, render_template, json, jsonify, request, Response
import io
import jsonpickle
import numpy as np
import cv2
from PIL import Image
from ModelClass import LgModel
from scripts.label_image import predict_result

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 mb image

@app.route("/")
def home():
    return render_template('Interior_design.html')

@app.route("/classify", methods=["POST"])
def get_image():
    file_object=request.files['photo']
    photo = file_object.read()
    #img = io.BytesIO(photo)
    #photo.save(in_memory_file,"PNG")
    #in_memory_file.seek(0)
    #photo=in_memory_file.read()
    #data=io.BytesIO(photo)
    #data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    #color_image_flag=-1
    #img = cv2.imdecode(data, color_image_flag)
    #model=LgModel()
    #prediction=model.predict(img)
    # do some fancy processing here....
    # encode response using jsonpickle
    #response_pickled = jsonpickle.encode(response)
    prediction=predict_result(photo)
    return Response(response=prediction, status=200, mimetype="text/plain")

# def get_filename():
#     user_data = request.get_json[]
#     n,r=int(user_data['n'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
