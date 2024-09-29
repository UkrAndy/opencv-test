from unittest.mock import right

import cv2
import math
import mediapipe as mp
import numpy as np
from constants.mediapipe_pose_labels import MediaPipePoseKeypoints
# from masc_test2 import landmarks
from utils.coordinates_calc import CoordinatesCalc


def dist_points(p1, p2):
    return  ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5

class PoseMasker:
    def __init__(self, joints_line_width =None, key_points_radius=None, head_width_line_ratio=1.5, head_width_radius_ratio=2):
        self.mp_pose = mp.solutions.pose
        self.joints_line_width = joints_line_width
        self.key_points_radius = key_points_radius
        self.head_width_line_ratio = head_width_line_ratio
        self.head_width_radius_ratio = head_width_radius_ratio

    def apply_mask_rects(self, image, landmarks, hands=['left', 'right']):
        image_width = image.shape[1]
        image_height = image.shape[0]

        # Створення чорного фону
        mask = np.zeros_like(image)

        left_ear = landmarks[MediaPipePoseKeypoints.LEFT_EAR.value ]
        right_ear = landmarks[MediaPipePoseKeypoints.RIGHT_EAR.value ]
        padding = math.fabs(right_ear.x - left_ear.x)

        right_shoulder = landmarks[MediaPipePoseKeypoints.RIGHT_SHOULDER.value ]
        left_shoulder = landmarks[MediaPipePoseKeypoints.LEFT_SHOULDER.value ]
        rect_width = math.fabs(left_shoulder.x - right_shoulder.x)

        right_heel = landmarks[MediaPipePoseKeypoints.RIGHT_HEEL.value ]
        nose = landmarks[MediaPipePoseKeypoints.NOSE.value ]
        rect_height = math.fabs(right_heel.y - nose.y)
        print('maxk -------- 1')
        # --- vertical rect ---
        v_top_left_x = int( max(right_shoulder.x - padding, 0) * image_width)
        v_top_left_y = int( max(nose.y - padding, 0) * image_height)

        v_bottom_right_x = int( min(left_shoulder.x + padding, 1) * image_width)
        v_bottom_right_y = int(min(right_heel.y + padding, 1) * image_height)

        cv2.rectangle(mask, (v_top_left_x, v_top_left_y),(v_bottom_right_x, v_bottom_right_y),
                      (255, 255, 255), -1)
        # --- horizontal rect ---
        right_wrist = landmarks[MediaPipePoseKeypoints.RIGHT_WRIST.value ]
        left_wrist = landmarks[MediaPipePoseKeypoints.LEFT_WRIST.value]
        hand_length = math.fabs(left_shoulder.x - right_shoulder.x)*2

        hand_wrist = right_wrist if hands[0] == 'right' else left_wrist

        print('maxk -------- 2')


        if hands[0] == 'right':
            h_top_left_x = int(max(
                min( math.fabs(right_wrist.x - padding), math.fabs(right_shoulder.x-hand_length)) ,
                0)* image_width)
        else:
            h_top_left_x = int(min(
                max( math.fabs(left_wrist.x + padding), math.fabs(left_shoulder.x+hand_length)) ,
                1)* image_width)

        h_top_left_y = int(max(
            min( math.fabs(nose.y - padding), math.fabs(hand_wrist.y - padding)),
            0) * image_height)

        if hands[0] == 'right':
            h_bottom_right_x = int(
                min( math.fabs(left_wrist.x + padding), 1) * image_width)
        else:
            h_bottom_right_x = int(min(
                max( math.fabs(right_wrist.x - padding), 0) ,
                1)* image_width)

        bottom_right_y = int(min( max(right_shoulder.y + hand_length, hand_wrist.y+padding), 1)* image_height)

        print('maxk -------- 3')


        cv2.rectangle(mask, (h_top_left_x, h_top_left_y), (h_bottom_right_x, bottom_right_y),
                      (255, 255, 255), -1)

        # Накладання маски на зображення
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def apply_mask(self, image, landmarks, hands=['left', 'right']):
        # Створення чорного фону
        mask = np.zeros_like(image)

        print('apply_mask landmarks')
        print(landmarks[0])


        # Обчислення ширини голови (відстань між вухами)
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        min_width = 50

        head_width = int(np.linalg.norm(np.array([left_ear.x, left_ear.y]) - np.array([right_ear.x, right_ear.y])) * image.shape[1])
        body_width = int(np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - np.array([right_shoulder.x, right_shoulder.y])) * image.shape[1])

        base_width = max(min_width, head_width)
        line_width =int(1.5*base_width)
        key_radius = int(0.4*base_width)

        head_width = max(base_width, body_width//2)



        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

        print('====>>> head_width =======================')
        print(head_width)



        # Відображення ключових точок та ліній між ними
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
            end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
            cv2.line(mask, start_point, end_point, (255, 255, 255), line_width)

        for landmark in landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(mask, (x, y), key_radius, (255, 255, 255), -1)
        for i in [6,3,18,20,19,17]:
            x = int(landmarks[i].x * image.shape[1])
            y = int(landmarks[i].y * image.shape[0])
            cv2.circle(mask, (x, y), key_radius, (255, 255, 255), -1)


        # Додавання прямокутників довкола долонь з padding
        hands_padding = int(base_width*0.6)
        if 'left' in hands:
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX.value]
            left_pinky = landmarks[self.mp_pose.PoseLandmark.LEFT_PINKY.value]
            left_thumb = landmarks[self.mp_pose.PoseLandmark.LEFT_THUMB.value]
            left_hand_bbox = [
                min(left_wrist.x, left_index.x, left_pinky.x, left_thumb.x) - hands_padding / image.shape[1],
                min(left_wrist.y, left_index.y, left_pinky.y, left_thumb.y) - hands_padding / image.shape[0],
                max(left_wrist.x, left_index.x, left_pinky.x, left_thumb.x) + hands_padding / image.shape[1],
                max(left_wrist.y, left_index.y, left_pinky.y, left_thumb.y) + hands_padding / image.shape[0]
            ]
            # left_hand_bbox = [
            #     min(left_wrist.x, left_index.x, left_pinky.x, left_thumb.x) ,
            #     min(left_wrist.y, left_index.y, left_pinky.y, left_thumb.y) ,
            #     max(left_wrist.x, left_index.x, left_pinky.x, left_thumb.x) ,
            #     max(left_wrist.y, left_index.y, left_pinky.y, left_thumb.y)
            # ]
            cv2.rectangle(mask, (int(left_hand_bbox[0] * image.shape[1]), int(left_hand_bbox[1] * image.shape[0])),
                          (int(left_hand_bbox[2] * image.shape[1]), int(left_hand_bbox[3] * image.shape[0])),
                          (255, 255, 255), head_width)

        if 'right' in hands:
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value]
            right_pinky = landmarks[self.mp_pose.PoseLandmark.RIGHT_PINKY.value]
            right_thumb = landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB.value]
            right_hand_bbox = [
                min(right_wrist.x, right_index.x, right_pinky.x, right_thumb.x) - hands_padding / image.shape[1],
                min(right_wrist.y, right_index.y, right_pinky.y, right_thumb.y) - hands_padding / image.shape[0],
                max(right_wrist.x, right_index.x, right_pinky.x, right_thumb.x) + hands_padding / image.shape[1],
                max(right_wrist.y, right_index.y, right_pinky.y, right_thumb.y) + hands_padding / image.shape[0]
            ]
            cv2.rectangle(mask, (int(right_hand_bbox[0] * image.shape[1]), int(right_hand_bbox[1] * image.shape[0])),
                          (int(right_hand_bbox[2] * image.shape[1]), int(right_hand_bbox[3] * image.shape[0])),
                          (255, 255, 255), head_width)
        # ----


        cv2.rectangle(mask, (int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])),
                      (int(left_hip.x * image.shape[1]), int(left_hip.y * image.shape[0])),
                      (255, 255, 255), -1)


        # Накладання маски на зображення
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    # -------------------------
    def get_palm_rect(self,pose, elbow, wrist, max_x, max_y):
        head_rad = int(dist_points(pose[4], pose[5]))
        body_rad = int(dist_points(pose[6], pose[7]) / 2)
        u_hand_length = dist_points(elbow, wrist)

        palm_width = int(
            max(head_rad, body_rad))  # , int(dist_points(elbow, wrist)/2), ) *(head_rad/u_hand_length/ratio))

        palm_direction = ((wrist[0] - elbow[0]) / u_hand_length, (wrist[1] - elbow[1]) / u_hand_length)
        palm_cx = int(wrist[0] + palm_direction[0] * palm_width * 3 / 4)
        palm_cy = int(wrist[1] + palm_direction[1] * palm_width * 3 / 4)
        return [
            [max(palm_cx - palm_width, 0), max(palm_cy - palm_width, 0)],
            [min(palm_cx + palm_width, max_x), min(palm_cy + palm_width, max_y)],
        ]
    # -------------------------
    def apply_mask_yolo(self, image, landmarks, hands=['left', 'right']):
        if landmarks is not None and len(landmarks)>0:
            # Створення чорного фону
            mask = np.zeros_like(image)

            # Відображення ключових точок та ліній між ними
            connections = [
                [0, 1],  # Ніс - Ліве око
                [0, 2],  # Ніс - Праве око
                [1, 3],  # Ліве око - Ліве вухо
                [2, 4],  # Праве око - Праве вухо
                [5, 6],  # Ліве плече - Праве плече
                [5, 7],  # Ліве плече - Лівий лікоть
                [7, 9],  # Лівий лікоть - Ліве зап'ястя
                [6, 8],  # Праве плече - Правий лікоть
                [8, 10],  # Правий лікоть - Праве зап'ястя
                [11, 12],  # Лівий стегно - Правий стегно
                [11, 13],  # Лівий стегно - Ліве коліно
                [13, 15],  # Ліве коліно - Ліва щиколотка
                [12, 14],  # Правий стегно - Праве коліно
                [14, 16]  # Праве коліно - Права щиколотка
            ]

            head_width =int(
                ((landmarks[16][0]-landmarks[17][0])**2+(landmarks[16][1]-landmarks[17][1])**2)**0.5
            )
            line_width =int(1*head_width)
            key_radius = int(0.4*head_width)

            for link in connections:
                start_idx = link[0]
                end_idx = link[1]

                start_point = [int(landmarks[start_idx][0]), int(landmarks[start_idx][1])]
                end_point = [int(landmarks[end_idx][0] ), int(landmarks[end_idx][1] )]

                cv2.line(mask, start_point, end_point, (255, 255, 255), line_width)

            for i in range(18):
                point = landmarks[i]
                cv2.circle(mask, [point[0], point[1]], key_radius, (255, 255, 255), -1)

            if 'left' in hands:
                rect = self.get_palm_rect(landmarks, landmarks[9], landmarks[11], len(image[0]), len(image))
                cv2.rectangle(mask, rect[0], rect[1], (255, 255, 255), thickness=-1)

            if 'right' in hands:
                rect = self.get_palm_rect(landmarks, landmarks[8], landmarks[10], len(image[0]), len(image))
                cv2.rectangle(mask, rect[0], rect[1], (255, 255, 255), thickness=-1)

            masked_image = cv2.bitwise_and(image, mask)
            return masked_image

    # ---------------------
    def convert_yolov7_to_mediapipe(self, yolov7_landmarks):
        yolov7_to_mediapipe = {
            0: self.mp_pose.PoseLandmark.NOSE,
            1: self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
            2: self.mp_pose.PoseLandmark.LEFT_EYE,
            3: self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
            4: self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            5: self.mp_pose.PoseLandmark.RIGHT_EYE,
            6: self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
            7: self.mp_pose.PoseLandmark.LEFT_EAR,
            8: self.mp_pose.PoseLandmark.RIGHT_EAR,
            9: self.mp_pose.PoseLandmark.MOUTH_LEFT,
            10: self.mp_pose.PoseLandmark.MOUTH_RIGHT,
            11: self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            12: self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            13: self.mp_pose.PoseLandmark.LEFT_ELBOW,
            14: self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            15: self.mp_pose.PoseLandmark.LEFT_WRIST,
            16: self.mp_pose.PoseLandmark.RIGHT_WRIST,
            17: self.mp_pose.PoseLandmark.LEFT_HIP,
            18: self.mp_pose.PoseLandmark.RIGHT_HIP,
            19: self.mp_pose.PoseLandmark.LEFT_KNEE,
            20: self.mp_pose.PoseLandmark.RIGHT_KNEE,
            21: self.mp_pose.PoseLandmark.LEFT_ANKLE,
            22: self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
        mediapipe_landmarks = [None] * len(self.mp_pose.PoseLandmark)

        for idx, yolo_idx in yolov7_to_mediapipe.items():
            mediapipe_landmarks[yolo_idx.value] = yolov7_landmarks[idx]

        return mediapipe_landmarks