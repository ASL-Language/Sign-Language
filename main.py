import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import psutil
import torch
import ctranslate2
import pyonmttok
from huggingface_hub import snapshot_download
from wordsegment import load, segment
from punctfix import PunctFixer
from spellchecker import SpellChecker
from concurrent.futures import ThreadPoolExecutor

class Recognizer:
    def __init__(self, model_path="./models/gesture_recognizer.task"):
        self.model_path = model_path
        self.recognizer = self._initialize_recognizer()

    def _initialize_recognizer(self):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

        options = GestureRecognizerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return GestureRecognizer.create_from_options(options)
    
    def recognize(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        result = self.recognizer.recognize(mp_image)
        return result

class HandMarker:
    def __init__(self, model_path="./models/hand_landmarker.task"):
        self.model_path = model_path
        self.hand_landmarker = self._initialize_hand_landmarker()

    def _initialize_hand_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            num_hands=2,
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )

        return HandLandmarker.create_from_options(options)

    def detect(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
        result = self.hand_landmarker.detect(mp_image)
        return result

    def draw_image(self, rgb_image, detection_result, gesture_result, text):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
        GESTURE_TEXT_COLOR = (0, 0, 255)  # red
        TEXT_VERTICAL_OFFSET = 30  # vertical offset between text lines

        height, width, _ = rgb_image.shape  # Ensure height and width are always assigned

        hand_landmarks_list = detection_result.hand_landmarks if detection_result else []
        handedness_list = detection_result.handedness if detection_result else []
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box.
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

            # Draw gesture if detected
            if gesture_result and gesture_result.gestures:
                gesture = gesture_result.gestures[idx][0]  # Get the gesture for the corresponding hand
                if gesture.score >= 0.85:
                    cv2.putText(
                        annotated_image,
                        f"Gesture: {gesture.category_name} ({gesture.score:.2f})",
                        (text_x, text_y + TEXT_VERTICAL_OFFSET),
                        cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE,
                        GESTURE_TEXT_COLOR,
                        FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

        # Draw text at the bottom of the image
        if text:
            cv2.putText(
                annotated_image,
                text,
                (MARGIN, height - MARGIN),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                GESTURE_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def gesture_to_text(self, gesture_result):
        if gesture_result and gesture_result.gestures:
            gestures = [
                f"{gesture.category_name}"
                for gesture in gesture_result.gestures[0]
                if gesture.score >= 0.85
            ]
            return " ".join(gestures)
        return ""

class OpenCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.time = 0
        self.cumulative_text = ""  # Variable to store cumulative text
        self.last_detection_time = time.time()  # Time of the last hand detection
        self.connect()

    def connect(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            exit()
        prev=""
        repeat_time=0
        while True:
            self.time += 1
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break

            # Detect hands synchronously
            hand_landmarker_result = self.detector.detect(frame)
            gesture_result = self.recognizer.recognize(frame) if hand_landmarker_result else None

            # Convert gesture result to text
            gesture_text = self.detector.gesture_to_text(gesture_result)

            # Translate text using the model
            if gesture_text :  # Ensure gesture_text is not empty
                if gesture_text==prev:
                    repeat_time+=1
                else:
                    repeat_time=0
                prev=gesture_text
                self.last_detection_time = time.time()
                if repeat_time==3:
                    if gesture_text=="space":
                        gesture_text=" "
                    self.cumulative_text+=gesture_text
                    prev=""
                    print(self.cumulative_text)
                    
                
            if self.cumulative_text:
                translated_text, _, _ = translate_batch_with_model([self.cumulative_text])
                self.cumulative_text = translated_text[0] if translated_text else ""
            # Clear cumulative text if no hand detected for 5 seconds
            if time.time() - self.last_detection_time > 5:
                self.cumulative_text = ""
                prev=""


            if hand_landmarker_result:
                # Draw landmarks and gestures on the frame
                mark_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_result, self.cumulative_text)
                cv2.imshow("Live", mark_image)
            else:
                cv2.imshow("Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 加载 wordsegment 数据
load()

# 初始化拼写检查器
spell = SpellChecker()

# 初始化 PunctFixer，只需要初始化一次
punct_fixer = PunctFixer()

# 初始化全局变量以缓存模型和分词器
ctranslate2_translator = None
pyonmttok_tokenizer = None

def process_text_with_model(text):
    # 使用 wordsegment 处理文本
    text = ' '.join(segment(text))

    # 使用 spellchecker 进行拼写纠正
    '''
    words = text.split()
    misspelled = spell.unknown(words)
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    text = ' '.join(corrected_words)
    '''
    return text

def initialize_model():
    global ctranslate2_translator, pyonmttok_tokenizer

    # 下载并缓存 CTranslate2 模型
    model_dir = snapshot_download(repo_id="jordimas/gec-opennmt-english", revision="main")

    # 初始化分词器
    pyonmttok_tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path=model_dir + "/sp_m.model")

    # 强制使用 CPU 而不是 GPU
    device = "cpu"

    # 初始化 CTranslate2 翻译器，启用多线程
    ctranslate2_translator = ctranslate2.Translator(model_dir, device=device, inter_threads=4, intra_threads=4)

def translate_batch_with_model(texts):
    global ctranslate2_translator, pyonmttok_tokenizer

    # 处理输入文本，使用多线程进行并行处理
    with ThreadPoolExecutor() as executor:
        processed_texts = list(executor.map(process_text_with_model, texts))

    # 测量开始时间和初始内存使用情况
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Tokenize and translate the processed texts
    '''
    tokenized_batch = [pyonmttok_tokenizer.tokenize(text)[0] for text in processed_texts]
    translated_batch = ctranslate2_translator.translate_batch(tokenized_batch)
    gec_corrected_texts = [pyonmttok_tokenizer.detokenize(translated.hypotheses[0]) for translated in translated_batch]
    '''
    # 使用 punctfix 处理标点符号，使用多线程进行并行处理
    with ThreadPoolExecutor() as executor:
        #punctuated_texts = list(executor.map(punct_fixer.punctuate, gec_corrected_texts))
        punctuated_texts = list(executor.map(punct_fixer.punctuate, processed_texts))
    # 测量结束时间和最终内存使用情况
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    # 计算时间和内存使用情况
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory

    return punctuated_texts, time_taken, memory_used

# 初始化模型
initialize_model()

# Example usage:
if __name__ == "__main__":
    OpenCamera()
