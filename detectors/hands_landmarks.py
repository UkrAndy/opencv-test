import mediapipe as mp
import copy
import itertools
import cv2 as cv


def calc_landmark_list(image_shape, landmarks):
    image_width, image_height = image_shape[0], image_shape[1]

    landmark_points = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_points.append([landmark_x, landmark_y])

    return landmark_points
# ------------
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_points in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_points[0], landmark_points[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
# -----------------

class HandsLandmark():
    def __init__(self, args, display_points=False):
        self.args = args
        self.display_points = display_points

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=args.use_static_image_mode,
            max_num_hands=args.max_num_hands,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        # ----------------
    def hands_landmark(self, image, hand_rect):
        hands_results = self.hands.process(image)
        if hands_results.multi_hand_landmarks is not None:
            try:
                result_list =[]
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks,
                                                      hands_results.multi_handedness):

                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = image.shape
                    wrist = [int(wrist.x * w), int(wrist.y * h)]


                    # Landmark calculation
                    # landmark_list = calc_landmark_list((self.args.width, self.args.height), hand_landmarks)
                    landmark_list = calc_landmark_list((w, h), hand_landmarks)


                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    result_list.append({'hand_landmark':pre_processed_landmark_list, 'wrist':wrist})

                if self.display_points:
                    self.draw_landmarks(image, landmark_list, hand_rect)


                return result_list
            except:
                return []
                print('Hands error')

    def draw_landmarks(self, image, landmark_points, hand_rect):
        if len(landmark_points) > 0:
            # Thumb
            cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_points):
            if index == 0:  # 手首1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手首2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image
