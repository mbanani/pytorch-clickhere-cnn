import numpy as np
import os
import sys
from scipy import linalg as linAlg


###############################################################################
######################### Paths to be set ####################################

# Angle conversions
def deg2rad(deg_angle):
    return deg_angle * np.pi / 180.0


def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    if deg_type == 'deg':
        xRot = deg2rad(xRot)
        yRot = deg2rad(yRot)
        zRot = deg2rad(zRot)

    xMat = np.array([
        [np.cos(xRot), np.sin(xRot), 0],
        [-np.sin(xRot), np.cos(xRot), 0],
        [0, 0, 1]
    ])

    yMat = np.array([
        [np.cos(yRot), 0, -np.sin(yRot)],
        [0, 1, 0],
        [np.sin(yRot), 0, np.cos(yRot)]
    ])

    zMat = np.array([
        [1, 0, 0],
        [0, np.cos(zRot), np.sin(zRot)],
        [0, -np.sin(zRot), np.cos(zRot)]
    ])

    return np.dot(zMat, np.dot(yMat, xMat))


def compute_angle_dists(preds, viewpoint_labels_as_mat):
    angle_dists = np.zeros(viewpoint_labels_as_mat.shape[0])
    for i in range(viewpoint_labels_as_mat.shape[0]):
        # Get rotation matrices from prediction and ground truth angles
        predR = angle2dcm(preds[i, 0], preds[i, 1], preds[i, 2])
        gtR = angle2dcm(viewpoint_labels_as_mat[i, 0], viewpoint_labels_as_mat[i, 1], viewpoint_labels_as_mat[i, 2])
        # Get geodesic distance
        angleDist = linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)
        angle_dists[i] = angleDist

    return angle_dists

def calculate_metrics(th0, th1, th2, object_types, labels, num_classes, threshold = np.pi/6):
    """ Function that calculates a set of metrics for viewpoint estimation
    Args:
        th0:            (N x C*360) array with all class predictions for azimuth
        th1:            (N x C*360) array with all class predictions for elevation
        th2:            (N x C*360) array with all class predictions for tilt
        object_types:   (N x 1) array with values ranging between (0, C-1)
        labels:         (N x 3) array with the ground-truth values for azim, elev, tilt
        threshold:      int setting acceptable accuracy threshold
        num_classes:    number of classes being predicted
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    # extract correct subset for the given class type
    theta_0 = extract_obj_prob(th0, object_types, num_classes)
    theta_1 = extract_obj_prob(th1, object_types, num_classes)
    theta_2 = extract_obj_prob(th2, object_types, num_classes)

    # convert from one-hot to int
    theta_0_ind = np.argmax(theta_0, axis=1) %360
    theta_1_ind = np.argmax(theta_1, axis=1) %360
    theta_2_ind = np.argmax(theta_2, axis=1) %360

    theta_0_ind = np.expand_dims(theta_0_ind, axis=1)
    theta_1_ind = np.expand_dims(theta_1_ind, axis=1)
    theta_2_ind = np.expand_dims(theta_2_ind, axis=1)

    # concatenate and convert from one-hot to normal labels
    prediction = np.concatenate([theta_0_ind, theta_1_ind, theta_2_ind], axis=1)

    # calculate the geodesic distance
    geo_dist = compute_angle_dists(prediction, labels)

    total_accuracy = 1.0 * np.mean(geo_dist <= threshold)
    total_mean_error = np.mean(geo_dist)

    # Per Type calculation
    all_correct  = geo_dist <= threshold
    type_geo_dist = [ [] for x in range(0, num_classes)]

    type_correct = np.zeros(num_classes, dtype=np.float16)
    type_total   = np.zeros(num_classes, dtype=np.float16)

    for i in range(0, object_types.shape[0]):
        type_correct[object_types[i]] += all_correct[i]
        type_total[object_types[i]]   += 1
        type_geo_dist[object_types[i]].append(geo_dist[i])

    type_total_altered = list(type_total)
    type_correct_altered = list(type_correct)

    for i in range(num_classes - 1, -1, -1):
        if type_total[i] == 0:
            del type_total_altered[i]
            del type_correct_altered[i]

    type_total_altered = np.asarray(type_total_altered)
    type_correct_altered = np.asarray(type_correct_altered)

    weighted_accuracy = np.mean(type_correct_altered/type_total_altered)

    return total_mean_error, total_accuracy, weighted_accuracy, type_correct, type_total, type_geo_dist, geo_dist


def extract_obj_prob(theta_full, obj_types, num_classes):
    """
        Function that extracts the probabilites associated with a specific object
        type from the overall prediction

        Args:
            theta_full:     Numpy array of size Nx360*C for azimuth ground-truth values
            obj_types:      Numpy array of size Nx1 for the specific object types
            num_classes:    Int specificing number of classes being considered
    """
    assert theta_full.shape[1] == num_classes*360, ("Incorrect number of objects calculated: label shape is %d, and expected is %d." % (theta_full.shape[1], num_classes*360))

    # Create an array of length batch_size * 360
    new_theta = np.ndarray([theta_full.shape[0], 360], dtype=theta_full.dtype)

    for i in range(0, theta_full.shape[0]):
        assert obj_types[i] >= 0 and obj_types[i] < num_classes
        assert theta_full.shape[1] == num_classes*360
        new_theta[i] = theta_full[i, 360*obj_types[i]:360*(obj_types[i]+1)]
    return new_theta

# TODO
# Future thought: Always create an evaluation class that maintains the results
# so that it's easy to extend it other statistics, as well as being able to
# show aggregate statistics over time, rather than having to do the messy work
# in the training loop
#
# Something to consider once things get working!
def evaluate_performance(azim_label, elev_label, tilt_label, azim_pr, elev_pr, tilt_pr, num_classes=12, obj_classes = None, threshold= np.pi/6):
    """
        Function that evaluates the performance of the model

        Args:
            azim_label: Numpy array of size Nx3 for azimuth ground-truth values
            elev_label: Numpy array of size Nx3 for elevation ground-truth values
            tilt_label: Numpy array of size Nx3 for tilt ground-truth values
            azim_pr:    Numpy array of size Nx(C*360) for azimuth predicted values
            elev_pr:    Numpy array of size Nx(C*360) for elevation predicted values
            tilt_pr:    Numpy array of size Nx(C*360) for tilt predicted values
            num_objs:   Number of classes being evaluated
    """
    obj_class = np.argmax(azim_label, axis=1)/360

    assert all(obj_classes == obj_class), "Error: Mismatch between extracted and set object categories"

    azim_gt = np.argmax(azim_label, axis=1)%360
    elev_gt = np.argmax(elev_label, axis=1)%360
    tilt_gt = (1 * np.argmax(tilt_label, axis=1))%360

    assert (np.argmax(azim_label, axis=1)/360 == np.argmax(elev_label, axis=1)/360).all()
    assert (np.argmax(azim_label, axis=1)/360 == np.argmax(tilt_label, axis=1)/360).all()

    azim_gt = np.expand_dims(azim_gt, axis=1)
    elev_gt = np.expand_dims(elev_gt, axis=1)
    tilt_gt = np.expand_dims(tilt_gt, axis=1)

    labels = np.concatenate([azim_gt, elev_gt, tilt_gt], axis=1)

    total_mean_error, total_accuracy, weighted_accuracy, type_correct, type_total, type_geo_dist, geo_dist = calculate_metrics(azim_pr, elev_pr, tilt_pr, obj_class, labels, threshold = threshold, num_classes=num_classes)

    return total_mean_error, total_accuracy, weighted_accuracy, type_correct, type_total, type_geo_dist, geo_dist


def label_to_probs(view_angles, object_class, flip, num_classes = 12):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by 360 * object_class_id + angle
    :return:
    '''
    # Calculate object multiplier
    if num_classes == 12:
        obj_mult = object_class
    else:
        print "Error: Invalid number of classes in data loader"

    # extract angles
    if flip:
        azim = np.mod(360-view_angles[0], 360)

        elev = view_angles[1] % 360

        tilt = np.mod(-1*view_angles[2], 360)
    else:
        azim = view_angles[0] % 360
        elev = view_angles[1] % 360
        tilt = view_angles[2] % 360

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
