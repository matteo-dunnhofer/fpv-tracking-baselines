import json
from got10k.trackers import Tracker
from got10k.utils.metrics import rect_iou
import numpy as np
from hic_detector import hic_config, hic_detect

class TbyDH(Tracker):

    def __init__(self):
        super(TbyDH, self).__init__(name='TbyD-H', is_deterministic=True)
        self.hic, self.cfg = hic_config()

    def init(self, image, box):
        self.prev_box = np.copy(box)

    def update(self, image):
        image = np.array(image)
        obj_preds = hic_detect(self.hic, self.cfg, image)
        obj_preds = np.array(obj_preds)

        if obj_preds.shape[0] > 0:
            prev_boxes = np.tile(np.array(self.prev_box), (obj_preds.shape[0], 1))
            ious = rect_iou(obj_preds, prev_boxes)
            
            new_box = obj_preds[np.argmax(ious)]

            if np.max(ious) <= 0.:
                new_box = self.prev_box
        else:
            new_box = self.prev_box

        self.prev_box = np.copy(new_box)  

        return new_box
