import numpy as np
import scipy.misc

from scipy import linalg as linAlg
from IPython import embed

def compute_angle_dists(preds, labels):
    # Get rotation matrices from prediction and ground truth angles
    predR   = angle2dcm(preds[0],  preds[1], preds[2])
    gtR     = angle2dcm(labels[0], labels[1], labels[2])

    # Get geodesic distance
    return linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)

def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    if deg_type == 'deg':
        xRot = xRot * np.pi / 180.0
        yRot = yRot * np.pi / 180.0
        zRot = zRot * np.pi / 180.0

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


class kp_dict(object):

    def __init__(self, num_classes = 12):
        self.keypoint_dict  = dict()
        self.num_classes    = num_classes
        self.class_ranges   = list(range(0, 360*(self.num_classes + 1), 360))
        self.threshold      = np.pi / 6.

    """
        Updates the keypoint dictionary
        params:     unique_id       unique id of each instance (NAME_objc#_kpc#)
                    predictions     the predictions for each vector
    """
    def update_dict(self, unique_id, predictions, labels):
        """Log a scalar variable."""
        if type(predictions) == int:
            predictions = [predictions]
            labels      = [labels]

        for i in range(0, len(unique_id)):
            image       = unique_id[i].split('_objc')[0]
            obj_class   = int(unique_id[i].split('_objc')[1].split('_kpc')[0])
            kp_class    = int(unique_id[i].split('_objc')[1].split('_kpc')[1])

            start_index = self.class_ranges[obj_class]
            end_index   = self.class_ranges[obj_class + 1]


            pred_probs = (  predictions[0][i], predictions[1][i], predictions[2][i])

            label_probs = ( labels[0][i], labels[1][i], labels[2][i])


            if image in list(self.keypoint_dict.keys()):
                self.keypoint_dict[image][kp_class] = pred_probs
            else:
                self.keypoint_dict[image] = {'class' : obj_class, 'label' : label_probs, kp_class : pred_probs}


    def calculate_geo_performance(self):
        for image in list(self.keypoint_dict.keys()):
            curr_label = self.keypoint_dict[image]['label']
            self.keypoint_dict[image]['geo_dist'] = dict()
            self.keypoint_dict[image]['correct'] = dict()
            for kp in list(self.keypoint_dict[image].keys()):
                if type(kp) != str :
                    curr_pred = [   np.argmax(self.keypoint_dict[image][kp][0]),
                                    np.argmax(self.keypoint_dict[image][kp][1]),
                                    np.argmax(self.keypoint_dict[image][kp][2])]
                    self.keypoint_dict[image]['geo_dist'][kp] = compute_angle_dists(curr_pred, curr_label)
                    self.keypoint_dict[image]['correct'][kp]  = 1 if (self.keypoint_dict[image]['geo_dist'][kp] <= self.threshold) else 0

    def metrics(self, unique = False):
        self.calculate_geo_performance()

        type_geo_dist   = [ [] for x in range(0, self.num_classes)]
        type_correct    = np.zeros(self.num_classes, dtype=np.float32)
        type_total      = np.zeros(self.num_classes, dtype=np.float32)

        for image in list(self.keypoint_dict.keys()):
            object_type = self.keypoint_dict[image]['class']
            curr_correct = 0.
            curr_total   = 0.
            curr_geodist = []
            for kp in list(self.keypoint_dict[image]['correct'].keys()):
                curr_correct += self.keypoint_dict[image]['correct'][kp]
                curr_total   += 1.
                curr_geodist.append(self.keypoint_dict[image]['geo_dist'][kp])

            if unique:
                curr_correct = curr_correct / curr_total
                curr_total   = 1.
                curr_geodist = [np.median(curr_geodist)]



            type_correct[object_type] += curr_correct
            type_total[object_type]   += curr_total
            for dist in curr_geodist:
                type_geo_dist[object_type].append(dist)

        type_accuracy   = np.zeros(self.num_classes, dtype=np.float16)
        for i in range(0, self.num_classes):
            if type_total[i] > 0:
                type_accuracy[i] = float(type_correct[i]) / type_total[i]

        self.calculate_performance_baselines()
        return type_accuracy, type_total, type_geo_dist


    def calculate_performance_baselines(self, mode = 'real'):

        mean_baseline       = [ [] for x in range(0, self.num_classes)]
        total_baseline      = [ [] for x in range(0, self.num_classes)]

        #iterate over batch
        for image in list(self.keypoint_dict.keys()):
            obj_cls = self.keypoint_dict[image]['class']

            perf = [self.keypoint_dict[image]['geo_dist'][kp] for kp in list(self.keypoint_dict[image]['geo_dist'].keys())]

            # Append baselines
            mean_baseline[obj_cls  ].append(np.mean(perf))
            for p in perf:
                total_baseline[obj_cls ].append(p )


        # embed()
        accuracy_mean    = np.around([ 100. * np.mean([ num < self.threshold for num in mean_baseline[i]   ]) for i in range(0, self.num_classes) ], decimals = 2)
        accuracy_total   = np.around([ 100. * np.mean([ num < self.threshold for num in total_baseline[i]  ]) for i in range(0, self.num_classes) ], decimals = 2)

        medError_mean    = np.around([ (180. / np.pi ) * np.median(mean_baseline[i]  ) for i in range(0, self.num_classes) ], decimals = 2)
        medError_total   = np.around([ (180. / np.pi ) * np.median(total_baseline[i] ) for i in range(0, self.num_classes) ], decimals = 2)


        if np.isnan(accuracy_mean[0]):
            accuracy_mean = accuracy_mean[[4,5,8]]
            accuracy_total = accuracy_total[[4,5,8]]
            medError_mean = medError_mean[[4,5,8]]
            medError_total = medError_total[[4,5,8]]

        print("--------------------------------------------")
        print("Accuracy ")
        print("mean      : ", accuracy_mean   , " -- mean : ", np.round(np.mean(accuracy_mean   ), decimals = 2))
        print("total     : ", accuracy_total  , " -- mean : ", np.round(np.mean(accuracy_total  ), decimals = 2))
        print("")
        print("Median Error ")
        print("mean      : ", medError_mean   , " -- mean : ",  np.round(np.mean(medError_mean   ), decimals = 2))
        print("total     : ", medError_total  , " -- mean : ",  np.round(np.mean(medError_total  ), decimals = 2))
        print("--------------------------------------------")
