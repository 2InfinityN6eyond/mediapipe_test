import numpy as np
import multiprocessing

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

class HolisticReader(multiprocessing.Process) :
    def __init__(self, to_main_process) :
        super(HolisticReader, self).__init__()

        self.to_main_process = to_main_process

    def run(self) :
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
            model_complexity = 2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                landmark_dict = {}
                if results.left_hand_landmarks :
                    left_hand_landmark_list = list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        results.left_hand_landmarks.landmark
                    )) 
                    landmark_dict["left_hand"] = left_hand_landmark_list
                if results.right_hand_landmarks :
                    right_hand_landmark_list = list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        results.right_hand_landmarks.landmark
                    ))
                    landmark_dict["right_hand"] = right_hand_landmark_list
                
                if results.pose_landmarks :
                    pose_landmark_list = list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        results.pose_landmarks.landmark
                    ))
                    landmark_dict["pose"] = pose_landmark_list
                
                if results.face_landmarks :
                    face_landmark_list = list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        results.face_landmarks.landmark
                    ))
                    landmark_dict["face"] = face_landmark_list

                self.to_main_process.put(landmark_dict)
            cap.release()


if __name__ == "__main__" :

    queue = multiprocessing.Queue
    holistic_reader = HolisticReader(queue)