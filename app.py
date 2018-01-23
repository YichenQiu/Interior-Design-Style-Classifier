from __future__ import division
from flask import Flask, render_template, request,jsonify
from flask_uploads import UploadSet, IMAGES, configure_uploads
from ModelClass import LgModel

app = Flask(__name__)
model = LgModel()
#photos = UploadSet('photos', IMAGES)
#app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
#configure_uploads(app, photos)

@app.route('/')
def index():
    return render_template('interior_design.html')

@app.route("/upload", methods=["POST"])
def upload():
    filename = request.form()
    predictions=model.predict(filename)
    return jsonify({'pred':predictions})



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
