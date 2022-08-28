from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from scripts.arm_entry import ArmEntry
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2


class Mouse_Detection(object):

    def __init__(self):
        self.score_threshold = 0.05
        self.top_k = 5

        self.config = "yolact_resnet50_mouse_config"
        self.trained_model = "./weights/yolact_v1_resnet50_mouse_123_10000.pth"

    def _order_port(self, list_port):
        min_x_index = np.argmin(
            [np.min(list_port[0], axis=0)[0], np.min(list_port[1], axis=0)[0], np.min(list_port[2], axis=0)[0]])
        min_y_index = np.argmin(
            [np.min(list_port[0], axis=0)[1], np.min(list_port[1], axis=0)[1], np.min(list_port[2], axis=0)[1]])
        max_y_index = np.argmax(
            [np.max(list_port[0], axis=0)[1], np.max(list_port[1], axis=0)[1], np.max(list_port[2], axis=0)[1]])
        list_result = [list_port[min_x_index], list_port[min_y_index], list_port[max_y_index]]
        return list_result

    def _get_mask(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb=False,
                            crop_masks=True,
                            score_threshold=self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            dic_mask = {}
            list_port = []
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]

            if len(classes) > 5:
                pass
            else:
                flag = 0
                for i, clas in enumerate(classes):
                    if clas == 0:
                        solutions = np.argwhere(masks[i] != 0)
                        list_port.append(solutions)
                        flag += 1
                        if flag == 3:
                            break
                    elif clas == 1:
                        solutions = np.argwhere(masks[i] != 0)
                        dic_mask['center'] = solutions
                    elif clas == 2:
                        solutions = np.argwhere(masks[i] != 0)
                        dic_mask['mouse'] = solutions

            if len(list_port) == 3:
                list_port_ = self._order_port(list_port)
                dic_mask['pA'] = list_port_[0]
                dic_mask['pC'] = list_port_[1]
                dic_mask['pB'] = list_port_[2]

        return dic_mask

    def _clear_list(self, list_label):
        list_result = [list_label[0]]
        i = j = 0
        while i < len(list_label) - 1:
            if list_label[i + 1] != list_result[j]:
                list_result.append(list_label[i + 1])
                i += 1
                j += 1
            else:
                i += 1
        return list_result

    def _log_video(self, net, video, flip=False):
        # If the path is a digit, parse it as a webcam index
        is_webcam = video.isdigit()
        list_result = []

        # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
        cudnn.benchmark = True

        if is_webcam:
            vid = cv2.VideoCapture(int(video))
        else:
            vid = cv2.VideoCapture(video)

        flag_start = 0
        flag_locate = "center"
        flag_center = True
        flag_port = False
        index = 0
        start_time = 0

        while True:
            ret, frame = vid.read()
            if not ret:
                cv2.waitKey(10)

                vid.release()
                cv2.destroyAllWindows()
                break

            # count the number of frames
            fps = vid.get(cv2.CAP_PROP_FPS)

            if flip:
                frame = cv2.flip(frame, 1)
            frame_flip = cv2.flip(frame, 1)
            frame_rotate = cv2.rotate(frame_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_rotate = torch.from_numpy(frame_rotate).cuda().float()
            batch = FastBaseTransform()(frame_rotate.unsqueeze(0))
            preds = net(batch)

            dic_mask = self._get_mask(preds, frame_rotate, None, None, undo_transform=False)
            arm_entry = ArmEntry(dic_mask)

            if flag_start == 0:
                if arm_entry.check_start():
                    start_time = index / fps
                    print("Start:", round(start_time, 2))
                    flag_start = 1
                    flag_center = True
                    flag_port = False
            else:
                if len(dic_mask) == 5:
                    if flag_center:
                        flag_locate, port = arm_entry.check_flag_center()
                        if port:
                            flag_center = False
                            flag_port = True
                    if flag_port:
                        flag_locate, center = arm_entry.check_flag_center()
                        if center:
                            flag_center = True
                            flag_port = False
                else:
                    continue

            if flag_locate != 'center':
                list_result.append(flag_locate)
            index += 1

            # calculate duration of the video
            end_time = index / fps - start_time
            if flag_start==1 and end_time >= 300:
                print("End:", round(end_time,2))
                break

        vid.release()
        cv2.destroyAllWindows()
        return self._clear_list(list_result)

    def _detection(self, net, video, flip=False):
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        return self._log_video(net, video, flip)

    def inference(self, video, flip=False):
        set_cfg(self.config)

        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            print('Loading model...', end='')
            net = Yolact()
            net.load_weights(self.trained_model)
            net.eval()
            print('Done.')
            net = net.cuda()

            return self._detection(net, video, flip)
