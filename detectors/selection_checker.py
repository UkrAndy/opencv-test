import time
from collections import defaultdict

class HandTracker():
    def __init__(self, max_no_select_id_count=2, select_id=0):
        self.timestamp = None
        self.max_no_select_id_count = max_no_select_id_count
        self.select_id = select_id
        self.clear_data()

    def clear_data(self):
        self.timestamp = None
        self.history =[]
        self.select_id_count = 0
        self.no_select_id_count = 0

    def is_selected_status(self):
        return  self.select_id_count>=(self.select_id_count+self.no_select_id_count)*0.9


    def add_data(self, hand_id):
        current_time = time.time()

        if hand_id == self.select_id:
            self.select_id_count += 1
            self.history.append(hand_id)
            if self.timestamp is None:
                self.timestamp = current_time

            elif current_time-self.timestamp >=3 :
                if self.is_selected_status():
                    self.clear_data()
                    return True
                else:
                    if self.history[0] == self.select_id:
                        self.select_id_count -= 1
                    else:
                        self.no_select_id_count -=1
                    self.history.pop(0)
        else:
            if self.timestamp is not None:
                self.no_select_id_count+=1
                if self.history[-1] != self.select_id or self.no_select_id_count>=self.max_no_select_id_count:
                    self.clear_data()
                else:
                    self.history.append(hand_id)
    def __str__(self):
        return  f"{self.select_id_count} - {self.no_select_id_count} - { self.history}"
# ---------------------------------------------


class HandsTracker:
    def __init__(self, max_no_select_id_count=2, select_id=0):
        self.trackers={
            'left': HandTracker(max_no_select_id_count=2, select_id=0),
            'right': HandTracker(max_no_select_id_count=2, select_id=0),
        }

    def clear_data(self):
        for key in self.trackers.keys():
            self.trackers[key].clear_data()

    def add_data(self, hands_data):
        if len(hands_data)>0:
            result = []
            not_used =['left', 'right']

            for hand_data in hands_data:
                if hand_data['hand_type'] in not_used:
                    not_used.remove(hand_data['hand_type'])
                if self.trackers[hand_data['hand_type']].add_data(hand_data['palm_id']):
                    result.append(hand_data['hand_type'])
            for key in not_used:
                self.trackers[key].clear_data()

            if len(result)>0:
                return result
            else:
                return None
        else:
            self.clear_data()
            return None