import numpy as np
import scipy.misc

from scipy import linalg as linAlg
from IPython import embed

class kp_dict(object):

    def __init__(self, num_classes = 12):
        self.keypoint_dict  = dict()
        self.num_classes    = num_classes
        self.class_ranges   = range(0, 360*(self.num_classes + 1), 360)
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


            pred_probs = (  predictions[0][i, start_index:end_index],
                            predictions[1][i, start_index:end_index],
                            predictions[2][i, start_index:end_index])

            label_probs = ( labels[0][i, start_index:end_index],
                            labels[1][i, start_index:end_index],
                            labels[2][i, start_index:end_index])


            if image in self.keypoint_dict.keys():
                self.keypoint_dict[image][kp_class] = pred_probs
            else:
                self.keypoint_dict[image] = {'class' : obj_class, 'label' : label_probs, kp_class : pred_probs}


    def calculate_geo_performance(self):
        for image in self.keypoint_dict.keys():
            curr_label = [  np.argmax(self.keypoint_dict[image]['label'][0]),
                            np.argmax(self.keypoint_dict[image]['label'][1]),
                            np.argmax(self.keypoint_dict[image]['label'][2])]
            self.keypoint_dict[image]['geo_dist'] = dict()
            self.keypoint_dict[image]['correct'] = dict()
            for kp in self.keypoint_dict[image].keys():
                if type(kp) != str :
                    curr_pred = [   np.argmax(self.keypoint_dict[image][kp][0]),
                                    np.argmax(self.keypoint_dict[image][kp][1]),
                                    np.argmax(self.keypoint_dict[image][kp][2])]
                    self.keypoint_dict[image]['geo_dist'][kp] = self.compute_angle_dists(curr_pred, curr_label)
                    self.keypoint_dict[image]['correct'][kp]  = 1 if (self.keypoint_dict[image]['geo_dist'][kp] < self.threshold) else 0

    def metrics(self, unique = False):
        self.calculate_geo_performance()

        type_geo_dist   = [ [] for x in range(0, self.num_classes)]
        type_correct    = np.zeros(self.num_classes, dtype=np.float32)
        type_total      = np.zeros(self.num_classes, dtype=np.float32)

        for image in self.keypoint_dict.keys():
            object_type = self.keypoint_dict[image]['class']
            curr_correct = 0.
            curr_total   = 0.
            curr_geodist = []
            for kp in self.keypoint_dict[image]['correct'].keys():
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


    def compute_angle_dists(self, preds, labels):
        # Get rotation matrices from prediction and ground truth angles
        predR   = self.angle2dcm(preds[0],  preds[1], preds[2])
        gtR     = self.angle2dcm(labels[0], labels[1], labels[2])

        # Get geodesic distance
        return linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)

    def angle2dcm(self, xRot, yRot, zRot, deg_type='deg'):
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


    def calculate_performance_baselines(self, mode = 'real'):

        worst_baseline  = [ [] for x in range(0, self.num_classes)]
        best_baseline   = [ [] for x in range(0, self.num_classes)]
        mean_baseline   = [ [] for x in range(0, self.num_classes)]
        median_baseline = [ [] for x in range(0, self.num_classes)]

        #iterate over batch
        for image in self.keypoint_dict.keys():
            obj_cls = self.keypoint_dict[image]['class']

            perf = [self.keypoint_dict[image]['geo_dist'][kp] for kp in self.keypoint_dict[image]['geo_dist'].keys()]

            best_baseline[obj_cls  ].append(np.min(perf))
            worst_baseline[obj_cls ].append(np.max(perf))
            mean_baseline[obj_cls  ].append(np.mean(perf))
            median_baseline[obj_cls].append(np.median(perf))

        accuracy_best    = np.around([ 100. * np.mean([ num < self.threshold for num in best_baseline[i]   ]) for i in range(0, self.num_classes) ], decimals = 2)
        accuracy_worst   = np.around([ 100. * np.mean([ num < self.threshold for num in worst_baseline[i]  ]) for i in range(0, self.num_classes) ], decimals = 2)
        accuracy_mean    = np.around([ 100. * np.mean([ num < self.threshold for num in mean_baseline[i]   ]) for i in range(0, self.num_classes) ], decimals = 2)
        accuracy_median  = np.around([ 100. * np.mean([ num < self.threshold for num in median_baseline[i] ]) for i in range(0, self.num_classes) ], decimals = 2)

        medError_best    = np.around([ (180. / np.pi ) * np.mean(best_baseline[i]  ) for i in range(0, self.num_classes) ], decimals = 2)
        medError_worst   = np.around([ (180. / np.pi ) * np.mean(worst_baseline[i] ) for i in range(0, self.num_classes) ], decimals = 2)
        medError_mean    = np.around([ (180. / np.pi ) * np.mean(mean_baseline[i]  ) for i in range(0, self.num_classes) ], decimals = 2)
        medError_median  = np.around([ (180. / np.pi ) * np.mean(median_baseline[i]) for i in range(0, self.num_classes) ], decimals = 2)

        print "--------------------------------------------"
        print "Accuracy "
        print "best      : ", accuracy_best   , " -- mean : ", np.round(np.mean(accuracy_best   ), decimals = 2)
        print "worst     : ", accuracy_worst  , " -- mean : ", np.round(np.mean(accuracy_worst  ), decimals = 2)
        print "mean      : ", accuracy_mean   , " -- mean : ", np.round(np.mean(accuracy_mean   ), decimals = 2)
        print "median    : ", accuracy_median , " -- mean : ", np.round(np.mean(accuracy_median ), decimals = 2)
        print "Median Error "
        print "best      : ", medError_best   , " -- mean : ",  np.round(np.mean(medError_best   ), decimals = 2)
        print "worst     : ", medError_worst  , " -- mean : ",  np.round(np.mean(medError_worst  ), decimals = 2)
        print "mean      : ", medError_mean   , " -- mean : ",  np.round(np.mean(medError_mean   ), decimals = 2)
        print "median    : ", medError_median , " -- mean : ",  np.round(np.mean(medError_median ), decimals = 2)
        print "--------------------------------------------"

class vp_dict(object):

    def __init__(self, num_classes = 12):

        self.threshold          = np.pi/6.
        self.num_classes        = num_classes

        self.class_ranges   = range(0, 360*12, 360)

        assert num_classes == len(self.class_ranges)

        self.results_class  = []
        self.results_pred   = []
        self.results_label  = []


    """
        Updates the keypoint dictionary
        params:     obj_class       object class                    (batch_size)
                    predictions     predictions for each image      (batch_size x 360*NumClasses)
                    labels          labels for each image           (batch_size x 360*NumClasses)
    """
    def update_dict(self, obj_class, predictions, labels):
        """Log a scalar variable."""
        if type(obj_class) == int:
            obj_class = [obj_class]
        if type(predictions) == int:
            predictions = [predictions]
        if type(labels) == int:
            labels = [labels]


        for i in range(0, len(obj_class)):
            start_index = self.class_ranges[obj_class[i]]
            self.results_class.append(obj_class[i])
            self.results_pred.append( [ np.argmax(predictions[0][i, start_index:start_index+360]),
                                        np.argmax(predictions[1][i, start_index:start_index+360]),
                                        np.argmax(predictions[2][i, start_index:start_index+360])])
            self.results_label.append( [ np.argmax(labels[0][i, start_index:start_index+360]),
                                         np.argmax(labels[1][i, start_index:start_index+360]),
                                         np.argmax(labels[2][i, start_index:start_index+360])])

    def metrics(self):
        geo_dist = self.compute_angle_dists(self.results_pred, self.results_label)

        # Per Type calculation
        all_correct  = geo_dist <= self.threshold
        type_geo_dist = [ [] for x in range(0, self.num_classes)]

        type_correct = np.zeros(self.num_classes, dtype=np.float32)
        type_total   = np.zeros(self.num_classes, dtype=np.float32)

        for i in range(0, len(self.results_class)):
            object_type = self.results_class[i]

            type_correct[object_type] += all_correct[i]
            type_total[object_type]   += 1.
            type_geo_dist[object_type].append(geo_dist[i])

        counter = 0.
        accuracy = 0.

        type_accuracy   = np.zeros(self.num_classes, dtype=np.float16)


        for i in range(0, self.num_classes):
            if type_total[i] > 0:
                type_accuracy[i] = float(type_correct[i]) / type_total[i]

        return type_accuracy, type_total, type_geo_dist


    def compute_angle_dists(self, preds, labels):
        angle_dists = np.zeros(len(labels))
        for i in range(0, len(labels)):
            # Get rotation matrices from prediction and ground truth angles
            predR   = self.angle2dcm(preds[i][0],  preds[i][1], preds[i][2])
            gtR     = self.angle2dcm(labels[i][0], labels[i][1], labels[i][2])

            # Get geodesic distance
            angleDist = linAlg.norm(linAlg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)
            angle_dists[i] = angleDist

        return angle_dists

    def angle2dcm(self, xRot, yRot, zRot, deg_type='deg'):
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
