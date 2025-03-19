from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import shutil
from predict import Predict
import os
import mimetypes

app = Flask(__name__, static_folder='./frontend/dist')
CORS(app)  # Enable CORS on all routes

# 初始化 Predict 类
model_path = "./final_model_state_dict_with_gate.pth"
predictor = Predict(model_path, device="cpu")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'zip_file' in request.files:
            zip_file = request.files['zip_file']

            if not os.path.exists("temp"):
                os.makedirs("temp")

            zip_path = os.path.join("temp", zip_file.filename)

            zip_path = zip_path.replace("\\", "/")

            zip_file.save(zip_path)

            results = predictor.predict(imgs=zip_path, mode="batch")

            os.remove(zip_path)

            return jsonify(results)

        elif 'left_eye' in request.files and 'right_eye' in request.files:
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

            results = predictor.predict(left_img=left_eye_path, right_img=right_eye_path, mode="single")

            os.remove(left_eye_path)
            os.remove(right_eye_path)

            return jsonify(results)

        else:
            return jsonify({'error': 'Both left_eye and right_eye files are required or a zip_file is required'}), 400
    except Exception as e:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        return jsonify({'error': str(e)}), 500


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