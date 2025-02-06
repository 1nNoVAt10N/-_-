from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from predict import Predict
import os
import mimetypes

app = Flask(__name__, static_folder='./frontend/dist')
CORS(app)  # Enable CORS on all routes

# 初始化 Predict 类
model_path = "./models/model_vit_1.pth"
predictor = Predict(model_path, device="cpu")

@app.route('/predict', methods=['POST'])
def predict():
    if 'left_eye' not in request.files or 'right_eye' not in request.files:
        return jsonify({'error': 'Both left_eye and right_eye files are required'}), 400

    left_eye_file = request.files['left_eye']
    right_eye_file = request.files['right_eye']

    if not os.path.exists("temp"):
        os.makedirs("temp")

    left_eye_path = os.path.join("temp", left_eye_file.filename)
    right_eye_path = os.path.join("temp", right_eye_file.filename)

    left_eye_path = left_eye_path.replace("\\", "/")
    right_eye_path = right_eye_path.replace("\\", "/")

    left_eye_file.save(left_eye_path)
    right_eye_file.save(right_eye_path)

    print(left_eye_path, right_eye_path)

    results = predictor.predict(left_img=left_eye_path, right_img=right_eye_path, mode="single")

    os.remove(left_eye_path)
    os.remove(right_eye_path)

    return jsonify(results)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_static_file(path)
    else:
        return send_static_file('index.html')

def send_static_file(f: str):
    if os.path.isfile(app.static_folder + '/' + f):
        if f.endswith('.js'):
            mime_type = 'text/javascript'
        else:
            mimetypes.init()
            mime_type, _ = mimetypes.guess_type(f)
        print(f, mime_type)
        return send_from_directory(app.static_folder, f, mimetype=mime_type)
    else:
        abort(404)

if __name__ == '__main__':
    app.run(debug=False, port=5000)