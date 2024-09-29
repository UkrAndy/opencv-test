#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2 as cv

from detectors.hands_landmarks import HandsLandmark
from detectors.gesture_landmarks import GestureDetector
from detectors.pose_landmarks import PoseDetector
from detectors.person_selector import PersonDetector
from utils.checkers.waiting_status_checker import get_open_hands, get_palms_gestures
from detectors.selection_checker import HandsTracker
from detectors.pose_mask import PoseMasker
from detectors.pose_calc_mp import PoseCalcMediapipe



from enum import Enum


class HandsStatus(Enum):
    OPEN = 0
    CLOSE = 1
    LUX = 2

class HandType(Enum):
    LEFT = 0
    RIGHT = 1


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default = 0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()
    return args

def draw_entire_pos(image, pose_points, args):
    if len(pose_points) > 0:
        i=0
        for point in pose_points:
            if 6<=i<=7 :
                cv.circle(image, [point[0], point[1]], 8, (255, 0, 0), 1)
            else:
                cv.circle(image, [point[0], point[1]], 8, (0, 255, 0), 1)
            i+=1
        cv.circle(image, [
            int((pose_points[6][0]+pose_points[7][0])/2),
            int((pose_points[6][1] + pose_points[7][1]) / 2)
        ], 8, (0, 0, 255), 2)

        cv.circle(image, [
            pose_points[10][0],pose_points[10][1]
        ], 8, (0, 0, 255), 2)

        cv.circle(image, [
            pose_points[11][0], pose_points[11][1]
        ], 8, (0, 0, 255), 2)

def draw_rectangles(image, rectangles):
    for rect in rectangles:
        region = rect['region']
        cv.rectangle(image, region[0], region[1], (0, 0, 255), 2)


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    args.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    args.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Model load #############################################################
    hands_getter = HandsLandmark(args, True)
    gesture_getter = GestureDetector()
    pose_detector = PoseDetector(args)
    person_selector = PersonDetector(args, True)
    hands_tracker = HandsTracker()
    pose_masker = PoseMasker()


    #  ########################################################################
    is_selected = False

    #  ########################################################################
    current_pose = None
    current_hand = None
    current_mask = None
    not_active_counter = 0
    max_not_active_counter = 3

    while True:
        #  ####################################################################
        try:
            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            print('============>>>> is_selected ===================')
            print(is_selected)

            if is_selected:
                try:
                    print('******** SELCTED ****************')
                    # current_pose = pose_detector.pose_landmark(image, True)
                    pose =  pose_detector.pose_landmark(current_mask, True)
                    if  pose is not None and len(pose)>1:
                        current_pose = pose
                        mask = pose_masker.apply_mask(image, current_pose, current_hand)
                        # mask = pose_masker.apply_mask_rects(image, current_pose, current_hand)
                        if current_mask is not None:
                            current_mask = mask
                            image = mask
                            if PoseCalcMediapipe.isSelectedHandActive(current_pose, current_hand, image, eps=0):
                                not_active_counter = 0
                            else:
                                not_active_counter += 1
                    else:
                        not_active_counter +=1
                    if not_active_counter > max_not_active_counter:
                        is_selected = False
                        current_pose = None
                        current_hand = None
                        current_mask = None
                except Exception as e:
                    print(f'stage2 error:  - main {e}')
                    print(current_pose)

            else:
                try:
                    print('*******  start person detect *********')
                    pose_selection = person_selector.select_person(image)

                    if pose_selection['pose_in_center'] is not None:
                        draw_entire_pos(image, pose_selection['pose_in_center'], args)
                        draw_rectangles(image, pose_selection['palms_rectangles'])

                        if len(pose_selection['palms_rectangles'])>0 :


                            open_hands = get_open_hands(image, pose_selection['palms_rectangles'],hands_getter, gesture_getter)
                            tracker_res = hands_tracker.add_data(open_hands)

                            if tracker_res is not None and len(tracker_res)>0:
                                # current_pose = pose_masker.convert_yolov7_to_mediapipe( pose_selection['pose_in_center'] )
                                current_mask = pose_masker.apply_mask_yolo(image, pose_selection['pose_in_center'])

                                if 'right' in tracker_res:
                                    current_hand = 'left'
                                else:
                                    current_hand = 'right'

                                not_active_counter = 0
                                is_selected = True
                                hands_tracker.clear_data()
                        else:
                            hands_tracker.clear_data()
                    else:
                        hands_tracker.clear_data()
                except Exception as e:
                    print(f'stage1 error:  - main {e}')

            cv.imshow('Hand Gesture Recognition', image)

            # Вихід при натисканні клавіші 'ESC'
            if cv.waitKey(10) == 27:  # ESC
                break



        except  Exception as e:
        #     # Якщо один з пайпів закритий і більше немає даних
            print(f'errrrrrrorrrr - main {e}')
            break

    cap.release()
    cv.destroyAllWindows()

# ==============================================

# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     landmark_array = np.empty((0, 2), int)
#
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#
#         landmark_point = [np.array((landmark_x, landmark_y))]
#
#         landmark_array = np.append(landmark_array, landmark_point, axis=0)
#
#     x, y, w, h = cv.boundingRect(landmark_array)
#
#     return [x, y, x + w, y + h]
#
#
# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     landmark_point = []
#
#     # Keypoint
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         # landmark_z = landmark.z
#
#         landmark_point.append([landmark_x, landmark_y])
#
#     return landmark_point
#
# def calc_landmark_list2(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     landmark_point = []
#
#     # Keypoint
#     for _, landmark in enumerate(landmarks.pose_landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         # landmark_z = landmark.z
#
#         landmark_point.append([landmark_x, landmark_y])
#
#     return landmark_point

# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)
#
#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]
#
#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
#
#     # Convert to a one-dimensional list
#     temp_landmark_list = list(
#         itertools.chain.from_iterable(temp_landmark_list))
#
#     # Normalization
#     max_value = max(list(map(abs, temp_landmark_list)))
#
#     def normalize_(n):
#         return n / max_value
#
#     temp_landmark_list = list(map(normalize_, temp_landmark_list))
#
#     return temp_landmark_list


# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]
#
#     temp_point_history = copy.deepcopy(point_history)
#
#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]
#
#         temp_point_history[index][0] = (temp_point_history[index][0] -
#                                         base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] -
#                                         base_y) / image_height
#
#     # Convert to a one-dimensional list
#     temp_point_history = list(
#         itertools.chain.from_iterable(temp_point_history))
#
#     return temp_point_history


# def logging_csv(number, mode, landmark_list, point_history_list):
#     if mode == 0:
#         pass
#     if mode == 1 and (0 <= number <= 9):
#         csv_path = 'model/keypoint_classifier/keypoint.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([number, *landmark_list])
#     if mode == 2 and (0 <= number <= 9):
#         csv_path = 'model/point_history_classifier/point_history.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([number, *point_history_list])
#     return


# def draw_landmarks(image, landmark_point):
#     if len(landmark_point) > 0:
#         # Thumb
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (255, 255, 255), 2)
#
#         # Index finger
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (255, 255, 255), 2)
#
#         # Middle finger
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (255, 255, 255), 2)
#
#         # Ring finger
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (255, 255, 255), 2)
#
#         # Little finger
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (255, 255, 255), 2)
#
#         # Palm
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (255, 255, 255), 2)
#
#     # Key Points
#     for index, landmark in enumerate(landmark_point):
#         if index == 0:  # 手首1
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 1:  # 手首2
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 2:  # 親指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 3:  # 親指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 4:  # 親指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 5:  # 人差指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 6:  # 人差指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 7:  # 人差指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 8:  # 人差指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 9:  # 中指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 10:  # 中指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 11:  # 中指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 13:  # 薬指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 14:  # 薬指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 15:  # 薬指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 16:  # 薬指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 17:  # 小指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 18:  # 小指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 19:  # 小指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 20:  # 小指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#
#     return image

# # -------------------
# def trim_pos(point, args):
#     res = list(point)
#     res[0]=max(0,res[0])
#     res[0]=min(res[0], args.width)
#     res[1]=max(0,res[1])
#     res[1]=min(res[1], args.height)
#     return res
#
# def draw_position(image, pose_points):
#     if len(pose_points) > 0:
#         if pose_points[16]:
#             point = trim_pos(pose_points[15])
#             cv.circle(image, point, 8, (0, 255, 0), 1)
#
#             # cv.circle(image, (landmark_point[0][0]+pose_points[15].x, landmark_point[0][1]+pose_points[15].y), 8, (0, 255, 0), 1)
#         if pose_points[16]:
#             point = trim_pos(pose_points[16])
#             cv.circle(image, point, 8, (0, 255, 0), 1)
#
#     return image

#
# def draw_bounding_rect(use_brect, image, brect):
#     if use_brect:
#         # Outer rectangle
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
#                      (0, 0, 0), 1)
#
#     return image


# def draw_info_text(image, brect, handedness, hand_sign_text):
#     cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)
#
#     info_text = handedness.classification[0].label[0:]
#     if hand_sign_text != "":
#         info_text = info_text + ':' + hand_sign_text
#     cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
#
#     # if finger_gesture_text != "":
#     #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#     #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
#     #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#     #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
#     #                cv.LINE_AA)
#
#     return image


# def draw_info(image, fps, mode, number):
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 0, 0), 4, cv.LINE_AA)
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (255, 255, 255), 2, cv.LINE_AA)
#
#     mode_string = ['Logging Key Point', 'Logging Point History']
#     if 1 <= mode <= 2:
#         cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                    cv.LINE_AA)
#         if 0 <= number <= 9:
#             cv.putText(image, "NUM:" + str(number), (10, 110),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                        cv.LINE_AA)
#     return image


if __name__ == '__main__':
    main()
