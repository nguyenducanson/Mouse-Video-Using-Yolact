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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--flip', default=False, type=str2bool, help='flip with input')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def order_port(list_port):
    min_x_index = np.argmin([np.min(list_port[0], axis=0)[0], np.min(list_port[1], axis=0)[0], np.min(list_port[2], axis=0)[0]])
    min_y_index = np.argmin([np.min(list_port[0], axis=0)[1], np.min(list_port[1], axis=0)[1], np.min(list_port[2], axis=0)[1]])
    max_y_index = np.argmax([np.max(list_port[0], axis=0)[1], np.max(list_port[1], axis=0)[1], np.max(list_port[2], axis=0)[1]])
    list_result = [list_port[min_x_index], list_port[min_y_index], list_port[max_y_index]]
    return list_result


def get_mask(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
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
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        dic_mask = {}
        list_port = []
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t]

        if len(classes) > 5: pass
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
            list_port_ = order_port(list_port)
            dic_mask['pA'] = list_port_[0]
            dic_mask['pC'] = list_port_[1]
            dic_mask['pB'] = list_port_[2]

    return dic_mask


def prep_display(img, max_score_label, dic_mask):
    if args.display_masks:
        for k, v in dic_mask.items():
            if k == 'mouse':
                new_image = cv2.polylines(img, [v], True, (0,0,255), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255), 2)
            elif k == 'center':
                new_image = cv2.polylines(img, [v], True, (0,255,255), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)
            else:
                new_image = cv2.polylines(img, [v], True, (0,255,0), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 2)

    new_image = cv2.putText(img, max_score_label, (200, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
    return new_image


def  get_outline(img, mask):
    dic_layout = {}
    black = np.ones(img.shape, dtype=np.uint8) * 0

    for k,v in mask.items():
        black_cp = black.copy()
        new_image = cv2.polylines(black_cp, [v], True, (0,0,255), 1)
        gray = cv2.cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = np.squeeze(np.array(cnts))
        dic_layout[k] = cnts.tolist()
        # print(cnts.shape)
        # print(type(cnts))
    # print(dic_layout)

    for k,v in dic_layout.items():
        # print(type(v))
        if k == 'mouse':
            cv2.polylines(black, [np.array(v)], True, (0,0,255), 1)
        elif k == 'center':
            cv2.polylines(black, [np.array(v)], True, (255,0,0), 1)
        else:
            cv2.polylines(black, [np.array(v)], True, (0,255,0), 1)
    return dic_layout, black


def clear_list(list_label):
    list_result = [list_label[0]]
    i = j = 0
    while i < len(list_label)-1:
        if list_label[i+1] != list_result[j]:
            list_result.append(list_label[i+1])
            i += 1
            j += 1
        else: i += 1
    return list_result


def evalimage(net:Yolact, path:str, save_path:str=None):
    frame = cv2.imread(path)
    if args.flip:
        frame = cv2.flip(frame, 1)
    frame_clip = cv2.flip(frame, 1)
    frame_rotate = cv2.rotate(frame_clip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_rotate = torch.from_numpy(frame_rotate).cuda().float()
    batch = FastBaseTransform()(frame_rotate.unsqueeze(0))
    preds = net(batch)

    img_numpy = frame

    dic_mask = get_mask(preds, frame_rotate, None, None, undo_transform=False)

    arm_entry = ArmEntry(dic_mask)

    if 'mouse' in dic_mask.keys() and 'center' in dic_mask.keys() and 'pA' in dic_mask.keys() and 'pB' in dic_mask.keys() and 'pC' in dic_mask.keys():
        img_numpy = prep_display(img_numpy, arm_entry.get_max_overlap(), dic_mask)

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def evalimages(net:Yolact, input_folder:str, output_folder:str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


def log_video(net:Yolact, path:str, out_path:str=None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    list_result = []
    
    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)
    flag_start = 0
    flag_locate = "center"
    flag_center = True
    flag_port = False
    index = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            cv2.waitKey(10)
            
            vid.release()
            cv2.destroyAllWindows()
            break

        if args.flip:
            frame = cv2.flip(frame, 1)
        frame_flip = cv2.flip(frame, 1)
        frame_rotate = cv2.rotate(frame_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rotate = torch.from_numpy(frame_rotate).cuda().float()
        batch = FastBaseTransform()(frame_rotate.unsqueeze(0))
        preds = net(batch)

        dic_mask = get_mask(preds, frame_rotate, None, None, undo_transform=False)
        arm_entry = ArmEntry(dic_mask)
        
        if flag_start == 0:
            if arm_entry.check_start():
                flag_start = 1
                flag_center = True
                flag_port = False
        else:
            if 'mouse' in dic_mask.keys() and 'center' in dic_mask.keys() and 'pA' in dic_mask.keys() and 'pB' in dic_mask.keys() and 'pC' in dic_mask.keys():
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

        img_numpy = frame
        img_numpy = prep_display(img_numpy, flag_locate, dic_mask)

        if out_path is not None:
            cv2.imwrite(f'{out_path}/{index}.jpg', img_numpy)
        index += 1

    vid.release()
    cv2.destroyAllWindows()
    return clear_list(list_result)


def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            print(log_video(net, inp, out))
        else:
            print(log_video(net, args.video))
        return


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None      

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)
