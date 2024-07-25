import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


class HandMarker:
    def __init__(self):
        self.model_path = "./models/hand_landmarker.task"
        self.hand_landmarker = self._initialize_hand_landmarker()
        self.frame = None

    def _initialize_hand_landmarker(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.print_result,
        )

        return HandLandmarker.create_from_options(options)

    def print_result(
        self,
        result: mp.tasks.vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        print("hand landmarker result: {}".format(result))
        if self.frame is not None:
            mark_image = self.drawImage(self.frame, result)
            cv2.imshow("Live", mark_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.end()

    def drawImage(self, rgb_image, detection_result):
        print("Drawing image...")  # Log message
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
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

        return annotated_image

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


class OpenCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandMarker()
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
            self.detector.frame = frame  # Store the current frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.detector.hand_landmarker.detect_async(
                mp_image, self.time
            )


# Example usage:
if __name__ == "__main__":
    OpenCamera()
