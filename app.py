from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import shutil
from predict import Predict
import os
import mimetypes
from data_preprocessing import PreprocessAndCache_for_single
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from cut_blend import cut_blend
app = Flask(__name__, static_folder='./frontend/dist')
CORS(app)  # Enable CORS on all routes

# 初始化 Predict 类
model_path = "./final_model_state_dict_with_gate.pth"
predictor = Predict(model_path, device="cpu")

@app.route('/predict', methods=['POST'])
def predict():
    left_eye_file = request.files['left_eye']
    right_eye_file = request.files['right_eye']
    left_eye_text = request.form.get('left_eye_text')
    right_eye_text = request.form.get('right_eye_text')
    patient_id = request.form.get('patientId')
    patient_name = request.form.get('patientName')
    patient_gender = request.form.get('patientGender')
    patient_age = request.form.get('patientAge')
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    left_eye_file_path = os.path.join("temp", left_eye_file.filename)
    right_eye_file_path = os.path.join("temp", right_eye_file.filename)

    left_eye_file_path = left_eye_file_path.replace("\\", "/")
    right_eye_file_path = right_eye_file_path.replace("\\", "/")

    left_eye_file.save(left_eye_file_path)
    right_eye_file.save(right_eye_file_path)

    pre=PreprocessAndCache_for_single(left_img="./"+left_eye_file_path, right_img="./"+right_eye_file_path,pre=True)
    img_left = pre.preprocess_img("./" + left_eye_file_path)
    img_right = pre.preprocess_img("./" + right_eye_file_path)
    img_left_x1,img_left_x2 = cut_blend("./"+left_eye_file_path)
    img_right_y1,img_right_y2 = cut_blend("./"+right_eye_file_path)
    # 将图像保存为 JPG 文件并转换为Base64编码
    img_left_x1_buffer = BytesIO()
    plt.imsave(img_left_x1_buffer, img_left_x1, format='jpg')
    img_left_x1_base64 = base64.b64encode(img_left_x1_buffer.getvalue()).decode('utf-8')

    img_left_x2_buffer = BytesIO()
    plt.imsave(img_left_x2_buffer, img_left_x2, format='jpg')
    img_left_x2_base64 = base64.b64encode(img_left_x2_buffer.getvalue()).decode('utf-8')

    img_right_y1_buffer = BytesIO()
    plt.imsave(img_right_y1_buffer, img_right_y1, format='jpg')
    img_right_y1_base64 = base64.b64encode(img_right_y1_buffer.getvalue()).decode('utf-8')


    img_right_y2_buffer = BytesIO()
    plt.imsave(img_right_y2_buffer, img_right_y2, format='jpg')
    img_right_y2_base64 = base64.b64encode(img_right_y2_buffer.getvalue()).decode('utf-8')
    result = predictor.predict(left_img="./"+left_eye_file_path, right_img="./"+right_eye_file_path,texts={
        "left_text": left_eye_text,
        "right_text": right_eye_text,
    },patient_id=patient_id,
    patrint_name=patient_name,
    patiend_gender=patient_gender,
    patiend_age=patient_age, mode="single")

    merged = pre.merge_double_imgs("./" + left_eye_file_path, "./" + right_eye_file_path)
    merged_buffer = BytesIO()
    plt.imsave(merged_buffer, merged, format='jpg')
    merged_base64 = base64.b64encode(merged_buffer.getvalue()).decode('utf-8')
    os.remove(left_eye_file_path)
    os.remove(right_eye_file_path)

    print(left_eye_file_path, right_eye_file_path)
    return jsonify({
        'merged_base64': merged_base64,
        'left_eye_x1': img_left_x1_base64,
        'left_eye_x2': img_left_x2_base64,
        'right_eye_y1': img_right_y1_base64,
        'right_eye_y2': img_right_y2_base64,
        'result': result
    }), 200

    # try:
    #     if 'zip_file' in request.files:
    #         zip_file = request.files['zip_file']

    #         if not os.path.exists("temp"):
    #             os.makedirs("temp")

    #         zip_path = os.path.join("temp", zip_file.filename)

    #         zip_path = zip_path.replace("\\", "/")

    #         zip_file.save(zip_path)

    #         results = predictor.predict(imgs=zip_path, mode="batch")

    #         os.remove(zip_path)

    #         return jsonify(results)

    #     elif 'left_eye' in request.files and 'right_eye' in request.files:
    #         left_eye_file = request.files['left_eye']
    #         right_eye_file = request.files['right_eye']

    #         if not os.path.exists("temp"):
    #             os.makedirs("temp")

    #         left_eye_path = os.path.join("temp", left_eye_file.filename)
    #         right_eye_path = os.path.join("temp", right_eye_file.filename)

    #         left_eye_path = left_eye_path.replace("\\", "/")
    #         right_eye_path = right_eye_path.replace("\\", "/")

    #         left_eye_file.save(left_eye_path)
    #         right_eye_file.save(right_eye_path)

    #         results = predictor.predict(left_img=left_eye_path, right_img=right_eye_path, mode="single")

    #         os.remove(left_eye_path)
    #         os.remove(right_eye_path)

    #         return jsonify(results)

    #     else:
    #         return jsonify({'error': 'Both left_eye and right_eye files are required or a zip_file is required'}), 400
    # except Exception as e:
    #     if os.path.exists("temp"):
    #         shutil.rmtree("temp")
    #     return jsonify({'error': str(e)}), 500


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
    app.run(debug=True, port=5000)