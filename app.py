from __future__ import division
from flask import Flask, render_template, request,jsonify
from ModelClass import LgModel

app = Flask(__name__)
model = LgModel()

@app.route('/')
def index():
    return render_template('interior_design.html')

@app.route("/classify", methods=["POST"])
def classify():
    user_data = request.get_json()

    predictions=model.predict(image)
    return jsonify({'pred':predictions})



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
