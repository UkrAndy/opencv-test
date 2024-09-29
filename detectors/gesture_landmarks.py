from model import KeyPointClassifier
import csv
import cv2 as cv

# ===========================================

class GestureDetector():
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1, csv_path='model/keypoint_classifier/keypoint_classifier_label.csv'):
        self.model_path = model_path
        self.num_threads = num_threads
        self.csv_path = csv_path

        # Model load #############################################################
        self.keypoint_classifier = KeyPointClassifier(model_path, num_threads)
    # ----------------

    # ----------------
    def gesture_landmark(self, hands_results_list, image, hand_data, display_info = False):
        if hands_results_list is not None:
            try:
                hand_sign_id_list=[]
                for pre_processed_landmark in hands_results_list:
                    # Hand sign classification
                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark)
                    hand_sign_id_list.append(hand_sign_id)
                    if display_info:

                        # Bounding box calculation
                        cv.rectangle(image, hand_data['region'][0], hand_data['region'][1], (255, 255, 255), 2)
                        info_text = hand_data['hand_type']
                        if hand_sign_id is not None:
                            info_text = info_text + ':' + str(hand_sign_id)


                        cv.putText(image, info_text, hand_data['region'][0],
                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


                return hand_sign_id_list

            except:
                return []
                print('Error gesture')
        else:
            return  None

    # ----------
    def read_labes(self):
        # Read labels ###########################################################
        with open(self.csv_path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
            return keypoint_classifier_labels
    # ----------------