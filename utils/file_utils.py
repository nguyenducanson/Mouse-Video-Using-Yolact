import os
import math
import datetime

import torch
import torch.nn as nn

from collections import deque
from pathlib import Path
from layers.interpolate import InterpolateModule


def check_path_exists(path):
    if os.path.exists(path):
        return True
    else:
        return False


def check_splitext_path(path):
    basename = os.path.basename(path)
    file_extension = os.path.splitext(basename)[1]
    return file_extension


def check_file_extension(path):
    mov = ['.mov', '.MOV']
    mp4 = ['.mp4', '.MP4']
    file_extension = check_splitext_path(path)
    if file_extension in mp4:
        return 'mp4'
    elif file_extension in mov:
        return 'mov'
    else:
        return 'NOT FIND FORMAT'
    
    
def calculate_sa_score(list_result: list):
    ''' aes: arm entries
        sa: spontaneous alterations
        sa_score = round(100*sa/(aes - 2), 2)
    '''
    aes = ''.join(list_result)
    sa = [aes[i:i+3] for i in range(len(aes)-2) if aes[i] != aes[i+1] and  aes[i+1] != aes[i+2] and aes[i+2] != aes[i] ]
    sa_score = round(100*len(sa)/(len(aes) - 2), 2)
    return aes, sa, sa_score


def save_result(video_dir):
    time = str(datetime.datetime.now())
    result_folder = os.path.join(video_dir, 'results')
    if not os.path.exists(result_folder):
      os.makedirs(os.path.join(video_dir, 'results'))
      
    csv_file = os.path.join(result_folder, str(time) + '.csv')
    return csv_file
