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

BUFFER_SIZE      = len(xfiles)
steps_per_epoch  = BUFFER_SIZE //BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch)


def load_images(xfile, yfile, flip=True):
    '''
    Lee par de imagenes jpeg y las reescala la tamaño deseado

    Aumantación: Flip horizontal aleatorio, sincronizado
    '''

    xim = tf.io.read_file(xfile)
    xim = tf.image.decode_bmp(xim)
    xim = tf.cast(xim, tf.float32)
    xim = xim / 127.5 - 1
    # en caso de ser necesario cambiar las dimensiones de la imagen x al leerla
    xim = tf.image.resize(xim, INPUT_DIM[:2],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    yim = tf.io.read_file(yfile)
    yim = tf.image.decode_bmp(yim)
    yim = tf.cast(yim, tf.float32)
    yim = yim / 127.5 - 1
    # en caso de ser necesario cambiar las dimensiones de la imagen y al leerla
    yim = tf.image.resize(yim, INPUT_DIM[:2],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Aumentación sincronizada de dada imágen del par $(x,y)$, en este caso solo un flip der-izq
    if flip and tf.random.uniform(()) > 0.5:
        xim = tf.image.flip_left_right(xim)
        yim = tf.image.flip_left_right(yim)

    return xim, yim


def display_images(fname, x_imgs=None, y_imgs=None, rows=4, cols=3, offset=0):
    '''
    Despliega pares de imágenes tomando una de cada lista
    '''
    plt.figure(figsize=(cols * 5, rows * 2.5))
    for i in range(rows * cols):
        plt.subplot(rows, cols * 2, 2 * i + 1)
        plt.imshow((x_imgs[i + offset] + 1) / 2)
        plt.axis('off')

        plt.subplot(rows, cols * 2, 2 * i + 2)
        plt.imshow((y_imgs[i + offset] + 1) / 2)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname)


rows = 2
cols = 2

x_imgs = []
y_imgs = []

for i in range(rows * cols):
    xim, yim = load_images(xfiles[i], yfiles[i])
    x_imgs.append(xim)
    y_imgs.append(yim)

print(x_imgs[0].shape, x_imgs[0].shape)  # a modo de comprobacion

display_images('testfigure1.png', x_imgs, y_imgs, rows=rows, cols=cols)

# Datasets configuration
idx = int(BUFFER_SIZE*.8)

train_x = tf.data.Dataset.list_files(xfiles[:idx], shuffle=False)
train_y = tf.data.Dataset.list_files(yfiles[:idx], shuffle=False)
train_xy = tf.data.Dataset.zip((train_x, train_y))
train_xy = train_xy.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
train_xy = train_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_xy = train_xy.batch(BATCH_SIZE)

test_x = tf.data.Dataset.list_files(xfiles[idx:], shuffle=False)
test_y = tf.data.Dataset.list_files(yfiles[idx:], shuffle=False)
test_xy = tf.data.Dataset.zip((test_x, test_y))
test_xy = test_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_xy = test_xy.batch(BATCH_SIZE)

rows=2
cols=2
for x,y in train_xy.take(1):
    display_images('testingdataset.png', x, y, rows=rows, cols=cols)
    break