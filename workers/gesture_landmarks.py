import mediapipe as mp
import copy
import itertools
from model import KeyPointClassifier




def gesture_getter(in_conn, out_conn, args):

    # Model load #############################################################
    keypoint_classifier = KeyPointClassifier()

    # ----------------
    while True:
        hands_results_list = in_conn[1].recv()
        # print('==== process gest start ======')
        if hands_results_list is not None:
            try:
                hand_sign_id_list=[]
                for pre_processed_landmark in hands_results_list:
                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark)
                    hand_sign_id_list.append(hand_sign_id)
                out_conn[0].send(hand_sign_id_list)
                # print('==== process gest end ======')

            except:
                out_conn[0].send([])
                print('Errorr gesture')