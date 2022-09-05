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

# UNet configuration
def downsample(filters, size, apply_batchnorm=True):
    '''
    Bloque de codificación (down-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters           = filters,
                                      kernel_size       = size,
                                      strides           = 2,
                                      padding           = 'same',
                                      kernel_initializer= initializer,
                                      use_bias          = False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model  = downsample(3, 4)
down_result = down_model(tf.expand_dims(xim, 0))
print(down_result.shape)

def upsample(filters, size, apply_dropout=False):
    '''
    Bloque de decodicación (up-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters           = filters,
                                               kernel_size       = size,
                                               strides           = 2,
                                               padding           = 'same',
                                               kernel_initializer= initializer,
                                               use_bias          = False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

# Generator model
def Generator():
    '''
    UNet
    '''

    # Capas que la componen
    x_input = tf.keras.layers.Input(shape=INPUT_DIM)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64,  64,  128)
        downsample(256, 4),  # (batch_size, 32,  32,  256)
        downsample(512, 4),  # (batch_size, 16,  16,  512)
        downsample(512, 4),  # (batch_size, 8,   8,   512)
        downsample(512, 4),  # (batch_size, 4,   4,   512)
        downsample(512, 4),  # (batch_size, 2,   2,   512)
        downsample(512, 4),  # (batch_size, 1,   1,   512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2,    2,  1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4,    4,  1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8,    8,  1024)
        upsample(512, 4),  # (batch_size, 16,   16, 1024)
        upsample(256, 4),  # (batch_size, 32,   32, 512)
        upsample(128, 4),  # (batch_size, 64,   64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    # pipeline de procesamiento
    x = x_input
    # Codificador
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)  # se agrega a una lista la salida cada vez que se desciende en el generador
    skips = reversed(skips[:-1])
    # Decodificador
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=x_input, outputs=x)
generator = Generator()


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(filters=1,
                                  kernel_size=4,
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss        = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss   = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss  = real_loss + generated_loss
    return total_disc_loss
LAMBDA = 100
def generator_loss(disc_generated_output, gen_output, target):
    '''
    el generador debe entrenarse para maximizar los errores de detección de imágenes sintéticas
    '''
    # Entropia cruzada a partir de logits
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Media de los Errores Absolutos
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
