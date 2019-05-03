import os
import numpy    as np
import scipy.io as spio

from IPython import embed

from util import Paths

INFO_FILE_HEADER = 'imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n'

synset_name_pairs = [   ('02691156', 'aeroplane'),
                        ('02834778', 'bicycle'),
                        ('02858304', 'boat'),
                        ('02876657', 'bottle'),
                        ('02924116', 'bus'),
                        ('02958343', 'car'),
                        ('03001627', 'chair'),
                        ('04379243', 'diningtable'),
                        ('03790512', 'motorbike'),
                        ('04256520', 'sofa'),
                        ('04468005', 'train'),
                        ('03211117', 'tvmonitor')]

KEYPOINT_TYPES = {
    'aeroplane'   : ['right_wing', 'tail', 'rudder_upper', 'noselanding',
                    'left_wing', 'rudder_lower', 'right_elevator', 'left_elevator'],
    'bicycle'     : ['left_front_wheel', 'left_back_wheel', 'seat_back',
                    'right_front_wheel', 'left_pedal_center', 'head_center',
                    'left_handle', 'right_pedal_center', 'right_handle',
                    'right_back_wheel', 'seat_front'],
    'boat'        : ['head', 'head_left', 'head_down', 'head_right',
                    'tail', 'tail_left', 'tail_right'],
    'bottle'      : ['body', 'body_right', 'body_left', 'bottom_right',
                    'bottom', 'mouth', 'bottom_left'],
    'bus'         : ['body_back_left_lower', 'body_back_left_upper', 'body_back_right_lower',
                    'body_back_right_upper', 'body_front_left_upper', 'body_front_right_upper',
                    'body_front_left_lower', 'body_front_right_lower', 'left_back_wheel',
                    'left_front_wheel', 'right_back_wheel', 'right_front_wheel'],
    'car'         : ['left_front_wheel', 'left_back_wheel', 'right_front_wheel',
                    'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield',
                    'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light',
                    'right_front_light', 'left_back_trunk', 'right_back_trunk'],
    'chair'       : ['seat_upper_right', 'leg_upper_left', 'seat_lower_left', 'leg_upper_right',
                    'back_upper_left', 'leg_lower_left', 'seat_upper_left', 'leg_lower_right',
                    'seat_lower_right', 'back_upper_right'],
    'diningtable' : ['top_upper_right', 'top_right', 'top_left', 'leg_upper_left',
                    'top_lower_left', 'top_lower_right', 'top_down', 'leg_upper_right',
                    'top_upper_left', 'leg_lower_left', 'leg_lower_right', 'top_up'],
    'motorbike'   : ['back_seat', 'front_seat', 'head_center', 'headlight_center',
                    'left_back_wheel', 'left_front_wheel', 'left_handle_center',
                    'right_back_wheel', 'right_front_wheel', 'right_handle_center'],
    'sofa'        : ['front_bottom_right', 'top_right_corner', 'top_left_corner',
                    'front_bottom_left', 'seat_bottom_right', 'seat_bottom_left',
                    'right_bottom_back', 'seat_top_right', 'left_bottom_back', 'seat_top_left'],
    'train'       : ['mid2_left_top', 'head_left_top', 'head_right_bottom', 'mid2_right_bottom',
                    'mid1_left_bottom', 'mid1_right_top', 'tail_right_bottom', 'tail_right_top', 'tail_left_top',
                    'head_right_top', 'head_left_bottom', 'mid2_left_bottom', 'tail_left_bottom', 'head_top',
                    'mid1_left_top', 'mid2_right_top', 'mid1_right_bottom'],
    'tvmonitor'   : ['front_bottom_right', 'back_top_left', 'front_top_left', 'front_bottom_left',
                    'back_bottom_left', 'front_top_right', 'back_top_right', 'back_bottom_right']
}


SYNSET_CLASSIDX_MAP = {}
for i in range(len(synset_name_pairs)):
    synset, _ = synset_name_pairs[i]
    SYNSET_CLASSIDX_MAP[synset] = i

KEYPOINT_CLASSES = []
for synset, class_name in synset_name_pairs:
    keypoint_names = KEYPOINT_TYPES[class_name]
    for keypoint_name in keypoint_names:
        KEYPOINT_CLASSES.append(class_name + '_' + keypoint_name)

KEYPOINTCLASS_INDEX_MAP = {}
for i in range(len(KEYPOINT_CLASSES)):
    KEYPOINTCLASS_INDEX_MAP[KEYPOINT_CLASSES[i]] = i

DATASET_SOURCES     = ['pascal', 'imagenet']
PASCAL3D_ROOT       = Paths.pascal3d_root
ANNOTATIONS_ROOT    = os.path.join(PASCAL3D_ROOT, 'Annotations')
IMAGES_ROOT         = os.path.join(PASCAL3D_ROOT, 'Images')


"""
    Create pascal image kp dataset for all classes.
    Code adapted from (https://github.com/rszeto/click-here-cnn)
"""
def create_pascal_image_kp_csvs(vehicles = False):
    # Generate train and test lists and store in file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.exists('trainImgIds.txt') and os.path.exists('valImgIds.txt')):
        matlab_cmd = 'addpath(\'%s\'); getPascalTrainVal' % BASE_DIR
        print('Generating MATLAB command: %s' % (matlab_cmd))
        os.system('matlab -nodisplay -r "try %s; catch; end; quit;"' % matlab_cmd)
        # os.system('matlab -nodisplay -r "%s; quit;"' % matlab_cmd)
        # Get training and test image IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
    with open('valImgIds.txt', 'rb') as testIdsFile:
        testIds = np.loadtxt(testIdsFile, dtype='string')

    data_dir = os.path.join(os.path.dirname(BASE_DIR), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    if vehicles:
        train_csv = os.path.join(data_dir, 'veh_pascal3d_kp_train.csv')
        valid_csv = os.path.join(data_dir, 'veh_pascal3d_kp_valid.csv')


        synset_name_pairs = [   ('02924116', 'bus'),
                                ('02958343', 'car'),
                                ('03790512', 'motorbike')]

    else:
        train_csv = os.path.join(data_dir, 'pascal3d_kp_train.csv')
        valid_csv = os.path.join(data_dir, 'pascal3d_kp_valid.csv')
        synset_name_pairs = [   ('02691156', 'aeroplane'),
                                ('02834778', 'bicycle'),
                                ('02858304', 'boat'),
                                ('02876657', 'bottle'),
                                ('02924116', 'bus'),
                                ('02958343', 'car'),
                                ('03001627', 'chair'),
                                ('04379243', 'diningtable'),
                                ('03790512', 'motorbike'),
                                ('04256520', 'sofa'),
                                ('04468005', 'train'),
                                ('03211117', 'tvmonitor')]

    SYNSET_CLASSIDX_MAP = {}
    for i in range(len(synset_name_pairs)):
        synset, _ = synset_name_pairs[i]
        SYNSET_CLASSIDX_MAP[synset] = i

    KEYPOINT_CLASSES = []
    for synset, class_name in synset_name_pairs:
        keypoint_names = KEYPOINT_TYPES[class_name]
        for keypoint_name in keypoint_names:
            KEYPOINT_CLASSES.append(class_name + '_' + keypoint_name)

    KEYPOINTCLASS_INDEX_MAP = {}
    for i in range(len(KEYPOINT_CLASSES)):
        KEYPOINTCLASS_INDEX_MAP[KEYPOINT_CLASSES[i]] = i

    info_file_train = open(train_csv, 'w')
    info_file_train.write(INFO_FILE_HEADER)
    info_file_test = open(valid_csv, 'w')
    info_file_test.write(INFO_FILE_HEADER)

    for synset, class_name in synset_name_pairs:
        print("Generating data for %s " % (class_name))
        all_zeros = 0
        counter = 0
        counter_kp = 0

        object_class = SYNSET_CLASSIDX_MAP[synset]
        for dataset_source in DATASET_SOURCES:
            class_source_id = '%s_%s' % (class_name, dataset_source)
            for anno_file in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, class_source_id))):
                anno_file_id = os.path.splitext(os.path.basename(anno_file))[0]
                if anno_file_id in trainIds:
                    anno_file_set = 'train'
                elif anno_file_id in testIds:
                    anno_file_set = 'test'
                else:
                    continue

                anno = loadmat(os.path.join(ANNOTATIONS_ROOT, class_source_id, anno_file))['record']
                rel_image_path = os.path.join('Images', class_source_id, anno['filename'])


                # Make objs an array regardless of how many objects there are
                objs = np.array([anno['objects']]) if isinstance(anno['objects'], dict) else anno['objects']
                for obj_i, obj in enumerate(objs):
                    # Only deal with objects in current class
                    if obj['class'] == class_name:
                        # Get crop using bounding box from annotation
                        # Note: Annotations are in MATLAB coordinates (1-indexed), inclusive
                        # Convert to 0-indexed numpy array
                        bbox = np.array(obj['bbox']) - 1

                        # Get visible and in-frame keypoints
                        keypoints = obj['anchors']
                        try:
                            assert set(KEYPOINT_TYPES[class_name]) == set(keypoints.keys())
                        except:
                            print("Assertion failed for keypoint types")
                            embed()

                        viewpoint = obj['viewpoint']
                        # Remove erronous KPs
                        if(viewpoint['azimuth'] == viewpoint['theta'] == viewpoint['elevation'] == 0.0):
                            all_zeros += 1
                        else:
                            counter += 1
                            azimuth = np.mod(np.round(viewpoint['azimuth']), 360)
                            elevation = np.mod(np.round(viewpoint['elevation']), 360)
                            tilt = np.mod(np.round(viewpoint['theta']), 360)

                            for keypoint_name in KEYPOINT_TYPES[class_name]:
                                # Get 0-indexed keypoint location
                                keypoint_loc_full = keypoints[keypoint_name]['location'] - 1
                                if keypoint_loc_full.size > 0 and insideBox(keypoint_loc_full, bbox):
                                    counter_kp += 1
                                    # Add info for current keypoint
                                    keypoint_class = KEYPOINTCLASS_INDEX_MAP[class_name + '_' + keypoint_name]
                                    if vehicles:
                                        if object_class == 0:
                                            final_label = ( 4, azimuth, elevation, tilt)
                                        elif object_class == 1:
                                            final_label = ( 5, azimuth, elevation, tilt)
                                        elif object_class == 2:
                                            final_label = ( 8, azimuth, elevation, tilt)
                                        else:
                                            print("Error: Object classes do not match expected values!")

                                    keypoint_str = keypointInfo2Str(rel_image_path, bbox, keypoint_loc_full, keypoint_class, final_label)
                                    if anno_file_set == 'train':
                                        info_file_train.write(keypoint_str)
                                    else:
                                        info_file_test.write(keypoint_str)
        print("%s : %d images %d image-kp pairs, %d ommited " % (class_name, counter, counter_kp, all_zeros))

    info_file_train.close()
    info_file_test.close()


"""
    Create pascal image kp dataset for all classes.
    Code adapted from (https://github.com/rszeto/click-here-cnn)
"""
def create_pascal_image_csvs(easy = False):
    # Generate train and test lists and store in file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.exists('trainImgIds.txt') and os.path.exists('valImgIds.txt')):
        matlab_cmd = 'addpath(\'%s\'); getPascalTrainVal' % BASE_DIR
        print('Generating MATLAB command: %s' % (matlab_cmd))
        os.system('matlab -nodisplay -r "try %s; catch; end; quit;"' % matlab_cmd)
        # os.system('matlab -nodisplay -r "%s; quit;"' % matlab_cmd)
        # Get training and test image IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
    with open('valImgIds.txt', 'rb') as testIdsFile:
        testIds = np.loadtxt(testIdsFile, dtype='string')

    data_dir = os.path.join(os.path.dirname(BASE_DIR), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if easy:
        train_csv = os.path.join(data_dir, 'pascal3d_train_easy.csv')
        valid_csv = os.path.join(data_dir, 'pascal3d_valid_easy.csv')
    else:
        train_csv = os.path.join(data_dir, 'pascal3d_train.csv')
        valid_csv = os.path.join(data_dir, 'pascal3d_valid.csv')


    info_file_train = open(train_csv, 'w')
    info_file_train.write(INFO_FILE_HEADER)
    info_file_test = open(valid_csv, 'w')
    info_file_test.write(INFO_FILE_HEADER)

    for synset, class_name in synset_name_pairs:
        print("Generating data for %s " % (class_name))
        all_zeros = 0
        hard_images = 0
        counter = 0
        object_class = SYNSET_CLASSIDX_MAP[synset]
        for dataset_source in DATASET_SOURCES:
            class_source_id = '%s_%s' % (class_name, dataset_source)
            for anno_file in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, class_source_id))):
                anno_file_id = os.path.splitext(os.path.basename(anno_file))[0]
                if anno_file_id in trainIds:
                    anno_file_set = 'train'
                elif anno_file_id in testIds:
                    anno_file_set = 'test'
                else:
                    continue

                anno = loadmat(os.path.join(ANNOTATIONS_ROOT, class_source_id, anno_file))['record']
                rel_image_path = os.path.join('Images', class_source_id, anno['filename'])

                # Make objs an array regardless of how many objects there are
                objs = np.array([anno['objects']]) if isinstance(anno['objects'], dict) else anno['objects']
                for obj_i, obj in enumerate(objs):
                    # Only deal with objects in current class
                    if obj['class'] == class_name:
                        # Get crop using bounding box from annotation
                        # Note: Annotations are in MATLAB coordinates (1-indexed), inclusive
                        # Convert to 0-indexed numpy array
                        bbox = np.array(obj['bbox']) - 1

                        viewpoint = obj['viewpoint']
                        # Remove erronous KPs
                        if(viewpoint['azimuth'] == viewpoint['theta'] == viewpoint['elevation'] == 0.0):
                            all_zeros += 1
                        elif (easy and (obj['difficult'] == 1 or obj['truncated'] == 1 or obj['occluded'] == 1 )):
                            hard_images += 1
                        else:
                            counter += 1
                            azimuth = np.mod(np.round(viewpoint['azimuth']), 360)
                            elevation = np.mod(np.round(viewpoint['elevation']), 360)
                            tilt = np.mod(np.round(viewpoint['theta']), 360)

                            final_label = ( object_class, azimuth, elevation, tilt)
                            viewpoint_str = viewpointInfo2Str(rel_image_path, bbox, final_label)
                            if anno_file_set == 'train':
                                info_file_train.write(viewpoint_str)
                            else:
                                info_file_test.write(viewpoint_str)
        print("%s : %d images, ommitted: all_zeros - %d, difficult - %d  " % (class_name, counter, all_zeros,hard_images))

    info_file_train.close()
    info_file_test.close()

######### Importing .mat files ###############################################
######### Reference: http://stackoverflow.com/a/8832212 ######################

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        # Handle case where elem is an array of mat_structs
        elif isinstance(elem, np.ndarray) and len(elem) > 0 and \
                isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
            dict[strg] = np.array([_todict(subelem) for subelem in elem])
        else:
            dict[strg] = elem
    return dict

def insideBox(point, box):
    return point[0] >= box[0] and point[0] <= box[2] \
            and point[1] >= box[1] and point[1] <= box[3]

def keypointInfo2Str(fullImagePath, bbox, keyptLoc, keyptClass, viewptLabel):
    return '%s,%d,%d,%d,%d,%f,%f,%d,%d,%d,%d,%d\n' % (
        fullImagePath,
        bbox[0], bbox[1], bbox[2], bbox[3],
        keyptLoc[0], keyptLoc[1],
        keyptClass,
        viewptLabel[0], viewptLabel[1], viewptLabel[2], viewptLabel[3]
    )

def viewpointInfo2Str(fullImagePath, bbox, viewptLabel):
    return '%s,%d,%d,%d,%d,%d,%d,%d,%d\n' % (
        fullImagePath,
        bbox[0], bbox[1], bbox[2], bbox[3],
        viewptLabel[0], viewptLabel[1], viewptLabel[2], viewptLabel[3]
    )

if __name__ == '__main__':
    # create_pascal_image_kp_csvs()
    create_pascal_image_kp_csvs(vehicles = True)       # Just vehicles
    # create_pascal_image_csvs()
    # create_pascal_image_csvs(easy = True)              # Easy subset
