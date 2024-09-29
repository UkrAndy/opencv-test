import cv2


def get_palm_points(image, region, palms_getter):
    fragment = image[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    palm_data = palms_getter.hands_landmark(fragment, region)

    return palm_data

def dist(p1,p2):
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
def get_true_coord(point, region):
    return [point[0]+region[0][0], point[1]+region[0][1]]

def get_one_true_hand_from_rectangle(image, region_data, palms_getter):
    palms_data = get_palm_points(image, region_data['region'], palms_getter)
    if palms_data is not None:
        if  len(palms_data)>1:
            min_d = 1e10

            for palm in palms_data:
                detected_wrist = get_true_coord(palm['wrist'], region_data['region'])

                cv2.circle(image,detected_wrist, 8,(255, 255, 255), 4)
                d = dist( detected_wrist , region_data['wrist_point'])
                if min_d > d:
                    min_d = d
                    true_palm = palm
            palms_data = [true_palm]

        else:
            detected_wrist = get_true_coord(palms_data[0]['wrist'], region_data['region'])

            cv2.circle(image, detected_wrist, 8, (255, 255, 255), 4)

    return palms_data




def get_palms_keypoints(image, palms_regions, palms_getter):
    palms_data=[]
    for region in palms_regions:
        palm_data = get_palm_points(image, region, palms_getter)
        if palm_data is not None:
            palms_data.append(palm_data)
    return  palms_data

# def get_correct_palm_data(region_data, palm_list):
    

def get_palms_gestures(image, palms_regions, palms_getter, gesture_getter, selection_palm_id=0):
    gestures_data=[]
    for region_data in palms_regions:
        # palm_data = get_palm_points(image, region_data['region'], palms_getter)
        palm_data = get_one_true_hand_from_rectangle(image, region_data, palms_getter)
        if palm_data is not None:
            palm_gesture_id = gesture_getter.gesture_landmark([palm_data[0]['hand_landmark']], image, region_data, True)

            if len(palm_gesture_id) == 1 and palm_gesture_id[0] == selection_palm_id:

                gestures_data.append({
                    'palm_id' : palm_gesture_id[0],
                    'hand_type':region_data['hand_type']
                })

    return  gestures_data

