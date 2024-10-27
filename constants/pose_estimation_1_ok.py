import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import multiprocessing
from hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from pose_estimation_pipeline import GStreamerPoseEstimationApp
from detectors.person_selector import PersonDetector


# -------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__(True)

     
# ===================================

class PointsOperation():
    @staticmethod
    def getAbsolutePoint(relativePoint, bbox,width,height):
        return [
            int((relativePoint.x() * bbox.width() + bbox.xmin()) * width),
            int((relativePoint.y() * bbox.height() + bbox.ymin()) * height)
        ]
    @staticmethod
    def getAbsolutePointsList(relativePointsList, bbox,width,height):
        return [
            PointsOperation.getAbsolutePoint(point, bbox,width,height)
            for point in relativePointsList
        ]
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    # string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Get the keypoints
    keypoints = get_keypoints()

    # print('**************  app_callback')
    persons =[]
    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            # string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            
            # Calculate the center of the bounding box
            center_x = width//2
            center_y = height//2

            # Draw a blue point at the center of the bounding box
            if user_data.use_frame:
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue point

            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                persons.append(PointsOperation.getAbsolutePointsList(points, bbox,width,height))
    
    PersonDetector.select_person ([user_data, frame, persons, width,height])
    return Gst.PadProbeReturn.OK

# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }

    return keypoints

if __name__ == "__main__":
    # queue_keypoints = multiprocessing.Queue(maxsize=5)
    # queue_center_pose = multiprocessing.Queue(maxsize=5)
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    # user_data.queue_keypoints = queue_keypoints
    
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    #----------------------
    # Create a queue for communication
        # Create producer and consumer processes
    # person_selector = PersonDetector(display_points = True)
    # person_selector_process = multiprocessing.Process(
    #     target=person_selector.select_person, 
    #     args=({
    #         'user_data':user_data
    #     },)
    # ) 
    # person_selector_process = multiprocessing.Process(
    #     target=person_selector.select_person, 
    #     args=({
    #         'getter':queue_keypoints,
    #         'setter':queue_center_pose
    #     },)
    # ) 

    # ---------------------
    # person_selector_process.start()   
    app.run()
    # person_selector_process.join()

