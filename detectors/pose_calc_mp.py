from constants.mediapipe_pose_labels import MediaPipePoseKeypoints
from utils.coordinates_calc import CoordinatesCalc
class PoseCalcMediapipe:
    # --- positions
    @staticmethod
    def isWristBelowElbow( elbow, wrist, eps=0):
        return wrist[1]>elbow[1] and CoordinatesCalc.dist_points(wrist, elbow)>eps
    @staticmethod
    def isWristAboveElbow( elbow, wrist, eps=0):
        return wrist[1]<elbow[1] and CoordinatesCalc.dist_points(wrist, elbow)>eps
    @staticmethod
    # --- width
    def getHeadWidth(current_pose):
        return int(CoordinatesCalc.dist_points(
            current_pose[MediaPipePoseKeypoints.LEFT_EAR.value],
            current_pose[MediaPipePoseKeypoints.RIGHT_EAR.value]
        ))

    def getBodyWidth(current_pose):
        return int(CoordinatesCalc.dist_points(
            current_pose[MediaPipePoseKeypoints.LEFT_SHOULDER.value],
            current_pose[MediaPipePoseKeypoints.RIGHT_SHOULDER.value]
        ))

    # --- status
    @staticmethod
    def isSelectedHandActive( current_pose, current_hand, image, rect=None, eps=None, relative_coords = True):

        if relative_coords:
            current_pose = CoordinatesCalc.relative_to_real_lendmark(image, current_pose, rect)


        if current_hand == 'left':
            elbow = current_pose[MediaPipePoseKeypoints.LEFT_ELBOW.value]
            wrist = current_pose[MediaPipePoseKeypoints.LEFT_WRIST.value]
        else:
            elbow = current_pose[MediaPipePoseKeypoints.RIGHT_ELBOW.value]
            wrist = current_pose[MediaPipePoseKeypoints.RIGHT_WRIST.value]

        if eps is None:
            eps = PoseCalcMediapipe.getHeadWidth(current_pose) / 2

        return  PoseCalcMediapipe.isWristAboveElbow(elbow,wrist,eps)

