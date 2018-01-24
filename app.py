from flask import Flask, render_template, json, jsonify, request, Response
import io
import jsonpickle
import numpy as np
import cv2
from ModelClass import LgModel

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32 mb image

@app.route("/")
def home():
    return render_template('Interior_design.html')

@app.route("/", methods=["POST"])
def get_image():
    photo = request.files['photo']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag=-1
    img = cv2.imdecode(data, color_image_flag)
    model=LgModel()
    prediction=model.predict(img)
    # do some fancy processing here....



    # encode response using jsonpickle
    #response_pickled = jsonpickle.encode(response)

    return Response(response=prediction, status=200, mimetype="text/plain")



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
