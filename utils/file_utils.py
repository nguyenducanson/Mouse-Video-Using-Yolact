import torch
import torch.nn as nn
import os
import math
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
