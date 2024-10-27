import cv2
import math
from detectors.pose_keypoints import Pose
import time

def dist(x1,y1,x2,y2):
    return  ((x2-x1)**2 + (y2-y1)**2)**0.5
def dist_points(p1, p2):
    return  ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
def dist_x(p1, p2):
    return  math.fabs(p1[0]-p2[0])



class PersonDetector():
    def __init__(self, args=None, display_points=False):
        self.args = args
        self.display_points = display_points

    @staticmethod
    def select_person(data, display_points = True):
        user_data, frame, persons, width,height = data
        cx = int(width/2)
        cy = int(height*0.6)
        # cy = int(height*3/4)
        # print(f'{cx} - {cy}')
        cv2.rectangle(frame, [0,0], [width,height], (255,0, 0), thickness=3)


        if display_points:
            # cv2.circle(frame, pose[1], 8, (0, 255, 255), 3)

            # cv2.circle(frame, point_left_shoulder, 8, (0, 255, 255), 3)
            # cv2.circle(frame, point_right_shoulder, 8, (0, 255, 255), 3)
            
            cv2.circle(frame, [
                int(cx), int(cy)
            ], 8, (255, 255, 0), 3)

        min_pose = None
        min_distance = 7e10
        palms_rectangles=[]
        mid_chest_point = None

        for pose in persons:
            # pose = []
            # for i in range(4, len(person), 3):
            #     x, y, conf = int(person[i]), int(person[i+1]), float(person[i+2])
            #     pose.append(([x, y, conf]))

            # pcx = (point_left_shoulder[0] + point_right_shoulder[0]+pose[12][0] + pose[13][0]) / 4
            # pcy = (point_left_shoulder[1] + point_right_shoulder[1]+pose[12][1] + pose[12][1]) / 4
            point_left_shoulder = pose[Pose.KEYPOINTS['left_shoulder']]
            point_right_shoulder = pose[Pose.KEYPOINTS['right_shoulder']]
            pcx = (point_left_shoulder[0] + point_right_shoulder[0]) // 2
            pcy = (point_left_shoulder[1] + point_right_shoulder[1]) // 2

            if display_points:
                # cv2.circle(frame, pose[1], 8, (0, 255, 255), 3)

                cv2.circle(frame, point_left_shoulder, 8, (0, 255, 255), 3)
                cv2.circle(frame, point_right_shoulder, 8, (0, 255, 255), 3)
                
                cv2.circle(frame, [
                    int(pcx), int(pcy)
                ], 8, (0, 255, 255), 3)


            dst = dist(cx,cy,pcx,pcy)
            if min_pose is None or (dist_x([pcx, pcy],[cx,cy])< dist_x(mid_chest_point,[cx,cy]) and dst < min_distance):
                min_distance = dst
                min_pose = pose

                point_left_shoulder = min_pose[Pose.KEYPOINTS['left_shoulder']]
                point_right_shoulder = min_pose[Pose.KEYPOINTS['right_shoulder']]
                pcx = (point_left_shoulder[0] + point_right_shoulder[0]) // 2
                pcy = (point_left_shoulder[1] + point_right_shoulder[1]) // 2
                mid_chest_point = [pcx, pcy]                
            # poses.append(pose)

        if min_pose is not None:
            eps = dist_points(min_pose[Pose.KEYPOINTS['left_ear']],min_pose[Pose.KEYPOINTS['right_ear']])/2
            if min_pose[Pose.KEYPOINTS['right_elbow']][1] > min_pose[Pose.KEYPOINTS['right_wrist']][1]+eps :
                palms_rectangles.append(
                    {
                        'hand_type':'right',
                        'region': PersonDetector.get_palm_rect(min_pose,min_pose[Pose.KEYPOINTS['right_elbow']],min_pose[Pose.KEYPOINTS['right_wrist']],width, height),
                        'wrist_point':min_pose[Pose.KEYPOINTS['right_wrist']]
                    }
                )

            if min_pose[Pose.KEYPOINTS['left_elbow']][1] > min_pose[Pose.KEYPOINTS['left_wrist']][1]+eps :            
                palms_rectangles.append(
                    {
                        'hand_type': 'left',
                        'region':PersonDetector.get_palm_rect(min_pose, min_pose[Pose.KEYPOINTS['left_elbow']],min_pose[Pose.KEYPOINTS['left_wrist']],width, height),
                        'wrist_point':min_pose[Pose.KEYPOINTS['left_wrist']]
                    }
                    )
            # print('=====>>>>>>>>    min_pose')
            # print(min_pose)


            # mid_chest_point = [int((min_point_left_shoulder[0] + min_point_right_shoulder[0]) / 2), int((min_point_left_shoulder[1] + min_point_right_shoulder[1]) / 2)]

            if display_points:
                if len(palms_rectangles)>=1:
                    for rect in palms_rectangles:
                       cv2.rectangle(frame, rect['region'][0], rect['region'][1], (0, 255, 0), thickness=3)
                # cv2.rectangle(frame, palms_rectangles[1][0], palms_rectangles[1][1], (0, 255, 0), thickness=3)
                cv2.circle(frame, mid_chest_point, 8, (255, 0, 0), 3)
                user_data.set_frame(frame)
        
        # print('ok')
        return  {
            'frame':frame,
            'poses':persons,
            'pose_in_center':min_pose,
            'palms_rectangles':palms_rectangles,
            'mid_chest_point':mid_chest_point
        }
        
        
        # cv2.imshow('Hand Gesture Recognition', frame)
        # key = cv2.waitKey(1)

        # time.sleep(1)

    @staticmethod
    def get_palm_rect(pose, elbow, wrist, max_x, max_y):
        point_left_shoulder = pose[Pose.KEYPOINTS['left_shoulder']]
        point_right_shoulder = pose[Pose.KEYPOINTS['right_shoulder']]

        head_rad = int(dist_points(pose[4], pose[5]))
        body_rad = int(dist_points(point_left_shoulder, point_right_shoulder) / 2)
        u_hand_length = dist_points(elbow, wrist)

        ratio = 3 / 2.5
        # palm_width = max(head_rad, body_rad, int(dist_points(elbow, wrist)/2), )
        palm_width = int(
            max(head_rad, body_rad))  # , int(dist_points(elbow, wrist)/2), ) *(head_rad/u_hand_length/ratio))

        palm_direction = ((wrist[0] - elbow[0]) / u_hand_length, (wrist[1] - elbow[1]) / u_hand_length)
        palm_cx = int(wrist[0] + palm_direction[0] * palm_width * 3 / 4)
        palm_cy = int(wrist[1] + palm_direction[1] * palm_width * 3 / 4)
        return [
            [max(palm_cx - palm_width, 0), max(palm_cy - palm_width, 0)],
            [min(palm_cx + palm_width, max_x), min(palm_cy + palm_width, max_y)],
        ]