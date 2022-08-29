from scripts.area_polygon import calculate_intersection_area


class ArmEntry:
    def __init__(self, dic_mask):
        self.dic_mask = dic_mask

    def check_start(self):
        if self._check_valid():
            list_score = self._get_overlap()  
            if list_score.index(max(list_score)) == 0 and max(list_score) != 0.0:
                return True
            return False
        return False

    def get_max_overlap(self):
        list_score = self._get_overlap()
        list_label = ['center', 'A', 'B', 'C']
        max_score = max(list_score)
        idx = list_score.index(max_score)
        return list_label[idx]

    def _get_overlap(self):
        overlap_mouse_center = calculate_intersection_area(self.dic_mask['mouse'], self.dic_mask['center'])
        overlap_mouse_pA = calculate_intersection_area(self.dic_mask['mouse'], self.dic_mask['pA'])
        overlap_mouse_pB = calculate_intersection_area(self.dic_mask['mouse'], self.dic_mask['pB'])
        overlap_mouse_pC = calculate_intersection_area(self.dic_mask['mouse'], self.dic_mask['pC'])
        return [overlap_mouse_center, overlap_mouse_pA, overlap_mouse_pB, overlap_mouse_pC]

    def check_flag_center(self):
        list_score = self._get_overlap()
        list_label = ['center', 'A', 'B', 'C']
        port = False
        if list_score[0] <= 0.9:
            port = True
            for i, score in enumerate(list_score[1:]):
                if score >= 0.1:
                    return list_label[i+1], port
        return list_label[0], port

    def check_flag_port(self):
        list_score = self._get_overlap()
        list_label = ['center', 'A', 'B', 'C']
        center = False
        if list_score[0] < 0.8:
            for i, score in enumerate(list_score[1:]):
                if score > 0.2:
                    return list_label[i+1], center
        else:
            center = True
            return list_label[0], center

    def _check_valid(self):
        if 'mouse' in self.dic_mask.keys() and 'center' in self.dic_mask.keys() and 'pA' in self.dic_mask.keys() and 'pB' in self.dic_mask.keys() and 'pC' in self.dic_mask.keys():
            return True
        return False