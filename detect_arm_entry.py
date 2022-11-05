import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import COCODetection, get_label_map
from data import cfg, set_cfg, set_dataset
from layers.output_utils import postprocess, undo_image_transformation
from left_right_classify.left_right_class import left_right
from scripts.arm_entry import ArmEntry
from utils import timer
from utils.augmentations import BaseTransform, FastBaseTransform
from utils.file_utils import check_path_exists, check_file_extension, calculate_sa_score, save_result, clear_list
from utils.functions import SavePath
from yolact import Yolact


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
                        default='weights/yolact_plus_resnet50_mouse_45_22500.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--count_time', default=300, type=int,
                        help='number of seconds to count arm entry')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--config', default='yolact_resnet50_mouse_config',
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--videos', default=None, type=str,
                        help='A path to a folder includes videos to evaluate on.')
    parser.add_argument('--score_threshold', default=0.05, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')

    parser.set_defaults(display=False, resume=False, output_coco_json=False, output_web_json=False,
                        benchmark=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def write_video(input, output, fps: int = 0, frame_size: tuple = (), ):
    if not fps or frame_size:
        vidcap = cv2.VideoCapture(input)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        _, arr = vidcap.read()
        height, width, _ = arr.shape
        frame_size = width, height
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        fps=fps,
        frameSize=frame_size
    )
    return writer


def check_flip(input, flip):
    vidcap = cv2.VideoCapture(input)
    _, arr = vidcap.read()
    left_right_object = left_right(arr)
    result = left_right_object.classify()

    if flip:
        result = not(result)

    return result


def order_port(list_port):
    min_x_index = np.argmin(
        [np.min(list_port[0], axis=0)[0], np.min(list_port[1], axis=0)[0], np.min(list_port[2], axis=0)[0]])
    min_y_index = np.argmin(
        [np.min(list_port[0], axis=0)[1], np.min(list_port[1], axis=0)[1], np.min(list_port[2], axis=0)[1]])
    max_y_index = np.argmax(
        [np.max(list_port[0], axis=0)[1], np.max(list_port[1], axis=0)[1], np.max(list_port[2], axis=0)[1]])
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
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        dic_mask = {}
        list_port = []
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        classes, _, _, masks = [x[idx].cpu().numpy() for x in t]

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
            list_port_ = order_port(list_port)
            dic_mask['pA'] = list_port_[0]
            dic_mask['pC'] = list_port_[1]
            dic_mask['pB'] = list_port_[2]

    return dic_mask


def prep_display(img, max_score_label, dic_mask):
    if args.display_masks:
        for k, v in dic_mask.items():
            if k == 'mouse':
                new_image = cv2.polylines(img, [v], True, (0, 0, 255), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 255), 2)
            elif k == 'center':
                new_image = cv2.polylines(img, [v], True, (0, 255, 255), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2)
            else:
                new_image = cv2.polylines(img, [v], True, (0, 255, 0), 1)
                cv2.putText(img, k, tuple(v[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 0), 2)

    new_image = cv2.putText(img, max_score_label, (200, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
    return new_image


def log_video(net: Yolact, path: str, out_path: str = None, csv_file: str = None, flip: bool = False, count_time: int = 300):
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
    start_time = 0

    # count the number of frames
    fps = vid.get(cv2.CAP_PROP_FPS)

    if out_path is not None:
        writer = write_video(path, out_path)

    if not check_flip(path, flip):
        flip = not (flip)

    while True:
        ret, frame = vid.read()
        if not ret:
            cv2.waitKey(10)

            vid.release()
            cv2.destroyAllWindows()
            break
        
        if flip:
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
                start_time = index / fps
                print("Start:", round(start_time, 2))
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
        # if flag_locate != 'center':
        list_result.append(flag_locate)

        if args.display:
            img_numpy = frame
            img_numpy = prep_display(img_numpy, flag_locate, dic_mask)

        if out_path is not None:
            writer.write(img_numpy)
            # cv2.imwrite(f'{out_path}/{index}.jpg', img_numpy)
        index += 1

        # calculate duration of the video
        end_time = index / fps - start_time

        if flag_start == 1 and end_time >= count_time:
            print("End:", round(end_time, 2))
            break

    vid.release()
    if out_path is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    aes, _, sa_score = calculate_sa_score(clear_list(list_result))
    
    if csv_file:
        with open(csv_file, 'a') as s:
            colnames = [
                "Video",
                "Start",
                "End",
                "AES",
                "SA Score"
            ]
            writer = csv.DictWriter(s, fieldnames=colnames)
            
            if os.stat(csv_file).st_size == 0:
                writer.writeheader()
            writer.writerow({
                "Video": path,
                "Start": str(round(start_time, 3)),
                "End": str(round(end_time, 3)),
                "AES": aes,
                "SA Score": str(round(sa_score, 3))
            })
            
        print(aes)
        print(f'Done with {path}')
        return
            
    return aes, sa_score


def evaluate(net: Yolact, dataset, train_mode=False, flip: bool = False, count_time: int = 300):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    
    if args.videos is not None:
        csv_file = save_result(args.videos)
        for d, _, files in os.walk(args.videos):
            for f in files:
                inp = os.path.join(d, f)
                if check_file_extension(f) == 'mov':
                    flip = not (flip)
                elif check_file_extension(f) == 'mp4':
                    flip = flip
                else:
                    continue
                    
                log_video(net, inp, csv_file=csv_file, flip=flip, count_time=count_time)
        return
               
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            if check_path_exists(inp):
                if check_file_extension(inp) == 'mov':
                    flip = not (flip)
                elif check_file_extension(inp) == 'mp4':
                    flip = flip
                else:
                    print('Not support')
                    return
            else:
                print('Not exits file')
                return
            
            print(log_video(net, inp, out, flip=flip, count_time=count_time))
        else:
            print(log_video(net, args.video, flip=flip, count_time=count_time))
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

        if args.videos is None and args.video is None:
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

        evaluate(net, dataset, count_time=args.count_time)
