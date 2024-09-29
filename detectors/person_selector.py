import cv2
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

def dist(x1,y1,x2,y2):
    return  ((x2-x1)**2 + (y2-y1)**2)**0.5
def dist_points(p1, p2):
    return  ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5




@torch.no_grad()
class PersonDetector():
    def __init__(self, args, display_points=False, poseweights = "yolov7-w6-pose.pt", source="0", device='cpu'):
        self.args = args
        self.display_points = display_points
        self.device = select_device(device)
        self.model = attempt_load(poseweights, map_location=self.device)
        self.model.eval()

    def select_person(self, frame):
        cx = int(len(frame[0])/2)
        cy = int(len(frame)*3/4)


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame.shape[1]), stride=64, auto=True)[0]
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_data, _ = self.model(image)

        output_data = non_max_suppression_kpt(output_data, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        keypoints = output_to_keypoint(output_data)

        min_pose = None
        min_distance = 7e10
        poses = []
        palms_rectangles=[]
        mid_chest_point = None

        for person in keypoints:
            pose = []
            for i in range(4, len(person), 3):
                x, y, conf = int(person[i]), int(person[i+1]), float(person[i+2])
                pose.append(([x, y, conf]))

            pcx = (pose[6][0] + pose[7][0]+pose[12][0] + pose[13][0]) / 4
            pcy = (pose[6][1] + pose[7][1]+pose[12][1] + pose[12][1]) / 4

            if self.display_points:
                cv2.circle(frame, [
                    int(pcx), int(pcy)
                ], 8, (0, 255, 255), 3)
                cv2.circle(frame, [
                    int(cx), int(cy)
                ], 8, (255, 255, 0), 3)

            dst = dist(cx,cy,pcx,pcy)
            if dst < min_distance:
                min_distance = dst
                min_pose = pose
            poses.append(pose)

        if min_pose is not None:
            eps = dist_points(min_pose[2],min_pose[3])/2
            if min_pose[8][1] > min_pose[10][1]+eps :
                palms_rectangles.append(
                    {
                        'hand_type':'right',
                        'region': self.get_palm_rect(min_pose,min_pose[8],min_pose[10],self.args.width, self.args.height),
                        'wrist_point':min_pose[10]
                    }
                )

            if min_pose[9][1] > min_pose[11][1] +eps:
                palms_rectangles.append(
                    {
                        'hand_type': 'left',
                        'region':self.get_palm_rect(min_pose,min_pose[9],min_pose[11],self.args.width, self.args.height),
                        'wrist_point':min_pose[11]
                    }
                    )

            mid_chest_point = [int((min_pose[6][0] + min_pose[7][0]) / 2), int((min_pose[6][1] + min_pose[7][1]) / 2)]

        return {
                'poses':poses,
                'pose_in_center':min_pose,
                'palms_rectangles':palms_rectangles,
                'mid_chest_point':mid_chest_point
                }

    def get_palm_rect(self,pose, elbow, wrist, max_x, max_y):
        head_rad = int(dist_points(pose[4], pose[5]))
        body_rad = int(dist_points(pose[6], pose[7]) / 2)
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