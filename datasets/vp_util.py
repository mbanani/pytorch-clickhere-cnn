import numpy as np
import os
import sys

def label_to_probs(view_angles, flip):
    # extract angles
    azim = view_angles[0] % 360
    elev = view_angles[1] % 360
    tilt = view_angles[2] % 360

    if flip:
        azim = (360-azim) % 360
        tilt = (-1 *tilt) % 360

