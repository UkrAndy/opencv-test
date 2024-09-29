import mediapipe as mp
import copy
import itertools


def calc_landmark_list(image_shape, landmarks):
    image_width, image_height = image_shape[0], image_shape[1]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point
# ------------
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

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

def hands_getter(in_conn, out_conn, args):
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    # ----------------

    while True:
        # print('==== process hands hands hands start ======')
        image = in_conn[1].recv()

        hands_results = hands.process(image)
        # print('==== process hands hands get ======')

        if hands_results.multi_hand_landmarks is not None:
            # print('==== process hands hands prosess 1 ======')
            try:
                result_list =[]
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks,
                                                      hands_results.multi_handedness):
                    # Landmark calculation
                    landmark_list = calc_landmark_list((args.width, args.height), hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    result_list.append(pre_processed_landmark_list)

                out_conn[0].send(result_list)
            except:
                out_conn[0].send([])
                print('Hands error')
        # print('==== process hands end ======')