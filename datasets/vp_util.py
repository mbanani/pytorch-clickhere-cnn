import numpy as np
import os
import sys
from scipy import linalg as linAlg
from IPython import embed

def label_to_probs(view_angles, object_class, flip, num_classes = 12):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by 360 * object_class_id + angle
    :return:
    '''
    # Calculate object multiplier
    obj_mult = object_class

    # extract angles
    azim = view_angles[0] % 360
    elev = view_angles[1] % 360
    tilt = view_angles[2] % 360

    if flip:
        # print ("Previous angle (%d, %d, %d) " % (azim, elev, tilt)),
        azim = (360-azim) % 360
        tilt = (-1 *tilt) % 360
        # print (". Flipped angle (%d, %d, %d) " % (azim, elev, tilt))

    # Loss parameters taken directly from Render4CNN paper
    azim_band_width = 7     # 15 in paper
    elev_band_width = 2     # 5 in paper
    tilt_band_width = 2     # 5 in paper

    azim_sigma = 5
    elev_sigma = 3
    tilt_sigma = 3

    azim_label = np.zeros((num_classes*360), dtype=np.float)
    elev_label = np.zeros((num_classes*360), dtype=np.float)
    tilt_label = np.zeros((num_classes*360), dtype=np.float)

    # calculate probabilities
    azim_band, azim_prob = calc_viewloss_vec(azim_band_width, azim_sigma)
    elev_band, elev_prob = calc_viewloss_vec(elev_band_width, elev_sigma)
    tilt_band, tilt_prob = calc_viewloss_vec(tilt_band_width, tilt_sigma)

    for i in azim_band:
        ind = np.mod(azim + i + 360, 360) + 360 * obj_mult
        azim_label[ind] = azim_prob[i + azim_band_width]

    for j in elev_band:
        ind = np.mod(elev + j + 360, 360) + 360 * obj_mult
        elev_label[ind] = elev_prob[j + elev_band_width]

    for k in tilt_band:
        ind = np.mod(tilt + k + 360, 360) + 360 * obj_mult
        tilt_label[ind] = tilt_prob[k + tilt_band_width]

    return azim_label, elev_label, tilt_label

def calc_viewloss_vec(size, sigma):
    band    = np.linspace(-1*size, size, 1 + 2*size, dtype=np.int16)
    vec     = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
    prob    = np.exp(-1 * abs(vec) / sigma)
    prob    = prob / np.sum(prob)

    return band, prob

if __name__ == '__main__':
    print "Nothing to run"
