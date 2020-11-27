from keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np

def tversky(y_true, y_pred, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    tp = K.sum(y_true_pos * y_pred_pos)
    fn = K.sum(y_true_pos * (1 - y_pred_pos))
    fp = K.sum((1 - y_true_pos) * y_pred_pos)

    return (tp + 1) / (tp + alpha * fn + (1 - alpha) * fp + 1)

def focal_tversky_loss(y_true, y_pred):
    gamma = .75  #todo default .75
    pt_1 = tversky(y_true, y_pred)

    return K.pow((1 - pt_1), gamma)

def tversky_loss(y_true, y_pred):
    loss = tversky(y_true, y_pred)

    return 1 - loss

def bce_dice_loss():
    def dsc_loss(y_true, y_pred):
        dice_loss = 1 - dsc(y_true, y_pred)
        bce_loss = binary_crossentropy(y_true, y_pred)

        return bce_loss + dice_loss

    return dsc_loss

def dice_loss(y_true, y_pred):

    return 1 - dsc(y_true, y_pred)

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score

def Active_Contour_Loss(y_true, y_pred):
    # y_pred = K.cast(y_pred, dtype = 'float64')

    """
    lenth term
    """

    x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2] ** 2
    delta_y = y[:, :, :-2, 1:] ** 2
    delta_u = K.abs(delta_x + delta_y)

    epsilon = 0.00000001  # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon))  # equ.(11) in the paper

    """
    region term
    """

    C_1 = np.ones((320, 1))
    C_2 = np.zeros((320, 1))

    region_in = K.abs(K.sum(y_pred[:, 0, :, :] * ((y_true[:, 0, :, :] - C_1) ** 2)))  # equ.(12) in the paper
    region_out = K.abs(K.sum((1 - y_pred[:, 0, :, :]) * ((y_true[:, 0, :, :] - C_2) ** 2)))  # equ.(12) in the paper

    lambdaP = 1  # lambda parameter could be various.

    loss = lenth + lambdaP * (region_in + region_out)

    return loss