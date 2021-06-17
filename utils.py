import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

########################################################################################################################
'''
MODEL UTILS:
'''
########################################################################################################################

def FFT_mag(input):
    real = input
    imag = tf.zeros_like(input)
    out = tf.abs(tf.signal.fft2d(tf.complex(real, imag)[:, :, 0]))
    return out

def model_loss(B1=1.0, B2=0.01):
    @tf.function
    def loss_func(y_true, y_pred):
        F_mag_true = tf.map_fn(FFT_mag, y_true)
        F_mag_pred = tf.map_fn(FFT_mag, y_pred)
        MAE_Loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
        F_mag_MAE_Loss = tf.keras.losses.MeanAbsoluteError()(F_mag_true, F_mag_pred)
        F_mag_MAE_Loss = tf.cast(F_mag_MAE_Loss, dtype=tf.float32)
        loss = B1*MAE_Loss + B2*F_mag_MAE_Loss
        return loss
    return loss_func

def normalize(tensor):

    return tf.math.divide_no_nan(tf.math.subtract(tensor, tf.math.reduce_min(tensor)), tf.math.subtract(tf.math.reduce_max(tensor), tf.math.reduce_min(tensor)))

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR

def SSIM(y_true, y_pred):
    max_pixel = 1.0
    y_true_norm = tf.map_fn(normalize, y_true)
    y_pred_norm = tf.map_fn(normalize, y_pred)
    SSIM = tf.image.ssim(y_true_norm,y_pred_norm,max_pixel,filter_size=11,
                         filter_sigma=1.5,k1=0.01,k2=0.03)
    return SSIM

def KLDivergence(y_true, y_pred):
    return tf.losses.KLDivergence()(y_true, y_pred)

def SavingMetric(y_true, y_pred):
    ssim = SSIM(y_true, y_pred)
    psnr = PSNR(y_true, y_pred)
    SSIM_norm = 1 - ssim
    PSNR_norm = (40 - psnr)/275
    loss = SSIM_norm + PSNR_norm
    return loss

def data_gen(filename):
    path = tf.keras.utils.get_file(filename,origin=None)
    with np.load(path) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images


def plot_history(history, metric, valmetric):
    offset = 0

    data1 = history.history[metric][offset:]
    data2 = history.history[valmetric][offset:]
    epochs = range(offset, len(data1) + offset)
    plt.plot(epochs, data1)
    plt.plot(epochs, data2)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(metric)
    plt.close()

def data_gen(filename):
    path = tf.keras.utils.get_file(filename,origin=None)
    with np.load(path) as data:
        src_images = data['arr_0']
        tar_images = data['arr_1']
    return src_images,tar_images

def plot_history(history,metric,valmetric):
    offset=0
    
    data1 = history.history[metric][offset:]
    data2 = history.history[valmetric][offset:]
    epochs = range(offset, len(data1) + offset)
    plt.plot(epochs, data1)
    plt.plot(epochs, data2)
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(metric)
    plt.close()

