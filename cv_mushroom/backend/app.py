from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# 初始化Flask
app = Flask(__name__)
CORS(app)  # 允许跨域请求
# 加载模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mushroom_model_final_optimized.h5')
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["可食用", "有毒"]  # 标签映射：0=可食用，1=有毒
# 训练时均值和标准差
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def preprocess_image(img):
    """图像预处理"""
    # 1. 转为float32并限制像素值范围
    img_array = image.img_to_array(img, dtype=np.float32)
    img_array = np.clip(img_array, 0.0, 255.0)

    # 2. 缩放到256x256，中心裁剪到224x224
    img_array = image.smart_resize(img_array, (256, 256))  # 先放大/缩小到256x256
    # 计算中心裁剪坐标
    h, w = img_array.shape[0], img_array.shape[1]
    start_h = (h - INPUT_SIZE[0]) // 2
    start_w = (w - INPUT_SIZE[1]) // 2
    img_array = img_array[start_h:start_h + INPUT_SIZE[0], start_w:start_w + INPUT_SIZE[1], :]

    # 3. 皈依化，标准化
    img_array = img_array / 255.0
    img_array = (img_array - RGB_MEAN) / RGB_STD

    # 4. 增加批次维度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "未上传图像"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "未选择图像"}), 400

        # 2. 读取图像
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"图像读取失败：{str(e)}"}), 400

        # 3. 预处理图像
        processed_img = preprocess_image(img)

        # 4. 模型预测
        prediction_prob = model.predict(processed_img, verbose=0)[0][0]  # 0-1区间的概率值
        # 确定类别
        predicted_class_idx = 1 if prediction_prob >= 0.5 else 0
        predicted_class = CLASS_NAMES[predicted_class_idx]
        # 计算置信度
        confidence = prediction_prob if predicted_class_idx == 1 else (1 - prediction_prob)
        confidence = round(confidence * 100, 2)

        # 5. 返回成功结果
        return jsonify({
            "status": "success",
            "result": {
                "class": predicted_class,
                "confidence": confidence,
            }
        })

    except Exception as e:
        # 捕获异常
        return jsonify({"error": f"预测失败：{str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)