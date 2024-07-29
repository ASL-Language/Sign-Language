import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

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

    def draw_image(self, rgb_image, detection_result, gesture_result):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
        GESTURE_TEXT_COLOR = (0, 0, 255)  # red
        TEXT_VERTICAL_OFFSET = 30  # vertical offset between text lines

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
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
            height, width, _ = annotated_image.shape
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

        return annotated_image

class OpenCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
        self.recognizer = Recognizer()
        self.time = 0
        self.connect()

    def connect(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            exit()
        while True:
            self.time += 1
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break

            # Detect hands synchronously
            hand_landmarker_result = self.detector.detect(frame)
            gesture_result = self.recognizer.recognize(frame) if hand_landmarker_result else None

            if hand_landmarker_result:
                # Draw landmarks and gestures on the frame
                mark_image = self.detector.draw_image(frame, hand_landmarker_result, gesture_result)
                cv2.imshow("Live", mark_image)
            else:
                cv2.imshow("Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    OpenCamera()
