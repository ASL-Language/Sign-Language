import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 打开默认摄像头（设备索引为 0
model = load_model("model/B_best_model_V1_3.keras")
cap = cv2.VideoCapture(0)
labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

# 加载训练好的模型
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while True:
    # 逐帧捕获
    ret, frame = cap.read()

    # 如果帧读取成功
    if ret:
        # 显示帧
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_frame = cv2.resize(gray_frame, (32, 32))
        input_frame = np.expand_dims(input_frame, axis=0)  # 增加批次维度
        input_frame = input_frame / 255.0  # 归一化
        predictions = model.predict(input_frame)
        print(predictions)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = labels[predicted_class]
        # 在帧上显示预测结果
        label = f"Class: {predicted_label}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
