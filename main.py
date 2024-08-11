import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
from collections import defaultdict
from wordsegment import load, segment
from punctfix import PunctFixer
from spellchecker import SpellChecker
import concurrent.futures
import psutil
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import ctranslate2
import pyonmttok
import re

ctranslate2_translator = None
pyonmttok_tokenizer = None

class Recognizer:
    def __init__(self, model_path="./models/gesture_recognizer-1_1.task"):
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

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

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

            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

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

        if gesture_result and gesture_result.gestures:
            gesture = gesture_result.gestures[0][0]  # Get the gesture for the corresponding hand
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
                if gesture.score >= 0.6
            ]
            return " ".join(gestures)
        return ""

class OpenCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.time = 0
        self.cumulative_text = ""
        self.last_detection_time = time.time()
        self.frame_counter = 0
        self.gesture_dict = defaultdict(int)  # Dictionary to store gesture counts
        self.window_size = 10  # Window size for moving average
        self.connect()

    def moving_average_filter(self):
        # 使用 numpy 計算滑動平均
        gestures = np.array(list(self.gesture_dict.values()))
        if gestures.size == 0:
            return None
        most_common_gesture_index = np.argmax(gestures)
        most_common_gesture = list(self.gesture_dict.keys())[most_common_gesture_index]
        return most_common_gesture

    def connect(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            exit()

        check_tran = 1
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break

            self.frame_counter += 1
            hand_landmarker_result = self.detector.detect(frame)
            gesture_result = self.recognizer.recognize(frame) if hand_landmarker_result else None
            gesture_text = self.detector.gesture_to_text(gesture_result)
            if gesture_text:
                self.gesture_dict[gesture_text] += 1  # Increment the count for this gesture
                self.last_detection_time=time.time()

            if self.frame_counter % self.window_size == 0:
                most_common_gesture = self.moving_average_filter()
                if most_common_gesture == "space":
                    most_common_gesture = " "
                self.cumulative_text += most_common_gesture if most_common_gesture else ""
                self.gesture_dict.clear()  # Reset the dictionary for the next window

                #print("Most common gesture in the last 5 frames:", most_common_gesture)
                #print("Cumulative text so far:", self.cumulative_text)
                #prev = most_common_gesture  # Update prev to track changes
            #print(time.time() - self.last_detection_time )
            # 新增：當偵測到是空白或時間差距大於3秒，丟給語言函式執行一次
            if self.cumulative_text and (most_common_gesture == " " or time.time() - self.last_detection_time > 3) and check_tran:
                self.cumulative_text = process_text_in_sequence(self.cumulative_text)
                check_tran = 0

            if hand_landmarker_result:
                mark_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_result, self.cumulative_text)
                cv2.imshow("Live", mark_image)
            else:
                cv2.imshow("Live", frame)

            if time.time() - self.last_detection_time > 5:
                self.cumulative_text = ""
                check_tran = 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


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


# 加载 wordsegment 数据
load()

# 初始化拼写检查器
spell = SpellChecker()

# 初始化 PunctFixer，只需要初始化一次
punct_fixer = PunctFixer()

#GEC
initialize_model()

def process_text_with_wordsegment(text):
    if text == "":
        return text
    return ' '.join(segment(text))

def process_text_with_spellchecker(text):
    if text == "":
        return text
    words = text.split()
    misspelled = spell.unknown(words)
    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]
    return ' '.join(corrected_words)

def process_text_with_punctfix(text):
    if text == "":
        return text
    return punct_fixer.punctuate(text)

def process_text_with_gec(text):
    """
    使用 GEC 模型进行校正。
    """
    # 分词
    tokenized_text = pyonmttok_tokenizer.tokenize(text)[0]

    # 翻译校正
    translated_batch = ctranslate2_translator.translate_batch([tokenized_text])
    gec_corrected_text = pyonmttok_tokenizer.detokenize(translated_batch[0].hypotheses[0])

    return gec_corrected_text

def remove_single_letters_except_ai(text):
    """
    移除句子中单独的字母，除了 'a' 和 'i' 之外。
    """
    # 使用正则表达式匹配单独的字母，前后可能有空格或标点符号
    text = re.sub(r'\b[b-hj-zB-HJ-Z]\b', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()  # 移除多余的空格

def safe_process(func, text):
    """
    安全地执行语言处理函数，确保不会返回 None。
    """
    result = func(text)
    return result if result is not None else text
#順序不保證
'''
def process_text_multithreaded(text):
    """
    使用多线程来同时运行语言处理函数，并在最后一步进行 GEC 校正。
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(safe_process, process_text_with_wordsegment, text): "wordsegment",
            executor.submit(safe_process, process_text_with_spellchecker, text): "spellchecker",
            executor.submit(safe_process, process_text_with_punctfix, text): "punctfix",
            executor.submit(safe_process, process_text_with_gec, text): "gec"
        }
        for future in concurrent.futures.as_completed(futures):
            text = future.result()
            print(f"{futures[future]}: {text}")
    
    return text
'''
def process_text_in_sequence(text):
    """
    顺序执行语言处理函数，确保按照指定顺序处理文本。
    """
    text = safe_process(process_text_with_wordsegment, text)
    print("wordsegment: ", text)

    text = safe_process(process_text_with_spellchecker, text)
    print("spellchecker: ", text)

    text = safe_process(process_text_with_punctfix, text)
    print("punctfix: ", text)

    text = safe_process(remove_single_letters_except_ai, text)
    print("after removing single letters: ", text)

    text = safe_process(process_text_with_gec, text)
    print("gec: ", text)
    
    return text
# Example usage:
if __name__ == "__main__":
    OpenCamera()
