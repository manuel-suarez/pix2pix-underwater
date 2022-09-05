import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

local_dir = '/home/est_posgrado_manuel.suarez/data/NYU_underwater'

INPUT_DIM     = (256,256,3)
OUTPUT_CHANNELS = INPUT_DIM[-1]
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0

xfiles  = glob(os.path.join(local_dir, '*_Underwater_*.bmp'))
yfiles  = glob(os.path.join(local_dir, '*_Image_*.bmp'))
xfiles.sort()
yfiles.sort()
xfiles=np.array(xfiles)
yfiles=np.array(yfiles)

[print(x, y) for x,y in zip(xfiles[:5],yfiles[:5])];