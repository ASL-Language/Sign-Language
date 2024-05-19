import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np

# 指定模型文件路径
model_path = "model/hand_landmarker.task"

# 创建 HandLandmarker 选项
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE,
)

# 初始化 HandLandmarker
with HandLandmarker.create_from_options(options) as hand_landmarker:
    # 读取图片
    image_path = "3.jpg"  # 修改为上传图片的路径
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图片包装为 Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # 运行手部检测模型
    hand_landmarker_result = hand_landmarker.detect(mp_image)

    # 如果检测到手部，绘制标记并抠图
    if hand_landmarker_result.hand_landmarks:
        for hand_landmarks in hand_landmarker_result.hand_landmarks:
            # 获取手部边界框
            x_min = min([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            x_max = max([landmark.x for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])

            # 添加缓冲区
            buffer = 0.05  # 缓冲区比例（可以调整）
            x_min = int((x_min - buffer) * image.shape[1])
            y_min = int((y_min - buffer) * image.shape[0])
            x_max = int((x_max + buffer) * image.shape[1])
            y_max = int((y_max + buffer) * image.shape[0])

            # 确保边界框在图像范围内
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, image.shape[1])
            y_max = min(y_max, image.shape[0])

            # 裁剪手部区域
            hand_image = image[y_min:y_max, x_min:x_max]

            # 创建透明背景
            hand_image_rgba = cv2.cvtColor(hand_image, cv2.COLOR_RGB2RGBA)

            # 遍历每个像素，将非手部区域的像素的 alpha 通道设为0（透明）
            for i in range(hand_image_rgba.shape[0]):
                for j in range(hand_image_rgba.shape[1]):
                    if (
                        hand_image_rgba[i, j, 0] == 0
                        and hand_image_rgba[i, j, 1] == 0
                        and hand_image_rgba[i, j, 2] == 0
                    ):
                        hand_image_rgba[i, j, 3] = 0  # 设置透明

            # 保存手部图像
            hand_image_path = f"hand.png"
            cv2.imwrite(hand_image_path, hand_image_rgba)

    print("Hand image saved at:", hand_image_path)
