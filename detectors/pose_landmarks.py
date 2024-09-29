import mediapipe as mp
import cv2

class PoseDetector():
    def __init__(self, args):
        self.args = args
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=args.use_static_image_mode,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def pose_landmark(self, image, display_points=False):
        try:
            results_pose = self.pose.process(image)
            if results_pose.pose_landmarks is not None and len(results_pose.pose_landmarks.landmark) > 0:
                res = list(results_pose.pose_landmarks.landmark)
                if display_points:
                    self.draw_entire_pos(image, res)
                return res
            else:
                return []
        except:
            return None

    def calc_pose_landmark_list(self, image, rel_points_list, pos_rect=None):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        # Keypoint
        for landmark in rel_points_list:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            if pos_rect is not None:
                landmark_x += pos_rect[0][0]
                landmark_y += pos_rect[0][1]

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def draw_entire_pos(self,image, relative_pose_points, pos_rect=None):
        if len(relative_pose_points) > 0:
            real_pose_points = self.calc_pose_landmark_list(image, relative_pose_points, pos_rect)
            for point in real_pose_points:
                cv2.circle(image, point, 8, (0, 255, 0), 1)


# def get_pose_getter(args):
#     # Model load #############################################################
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(
#         static_image_mode=args.use_static_image_mode,
#         min_detection_confidence=args.min_detection_confidence,
#         min_tracking_confidence=args.min_tracking_confidence,
#     )
#     # ----------------
#     def pose_landmark(image):
#         try:
#             results_pose = pose.process(image)
#             if results_pose.pose_landmarks is not None and len(results_pose.pose_landmarks.landmark) > 0:
#                 return list(results_pose.pose_landmarks.landmark)
#             else:
#                 return []
#         except:
#             return None
#
#     return  pose_landmark

