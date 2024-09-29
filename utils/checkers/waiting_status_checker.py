from utils.checkers.hands_status_checker import get_palms_keypoints, get_palms_gestures

def get_open_hands(image,  palms_regions, palms_getter, gesture_getter, selection_palm_id=0):
    pals_gestures = get_palms_gestures(image,  palms_regions, palms_getter,gesture_getter, selection_palm_id)
    return pals_gestures