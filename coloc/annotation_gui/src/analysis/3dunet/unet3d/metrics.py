from functools import partial

from keras import backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


# def tversky_coefficient(y_true, y_pred, alpha=0.6, smooth=1):
#    p1 = 1 - y_pred
#    g1 = 1 - y_true

#    numerator = K.sum(y_pred * y_true, -1)
#    denominator = K.sum(y_pred * y_true + alpha * y_pred * g1 + (1 - alpha) * p1 * y_true, -1)

#    return (numerator + smooth)/(denominator + smooth)

def tversky_coefficient(y_true, y_pred, beta=0.5, smooth=1, axis=(-1)):
    numerator = tf.reduce_sum(y_true * y_pred, axis=axis)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    tc = (numerator + smooth)/(K.sum(denominator, axis=axis) + smooth)
    return tc


def tversky_loss(y_true, y_pred):
    return -tversky_coefficient(y_true, y_pred)


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
