import torch
import time
import copy
import random
import pandas
import os

import numpy    as np

from PIL            import Image
from .vp_util        import label_to_probs
from torchvision    import transforms
from IPython        import embed


class pascal3d_kp(torch.utils.data.Dataset):

    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, csv_path, dataset_root = None, im_size = 227, transform = None, map_size = 46, num_classes = 12, flip = False):

        assert transform           != None

        start_time = time.time()

        # Load instance data from csv-file
        im_paths, bbox, kp_loc, kp_cls, obj_cls, vp_labels = self.csv_to_instances(csv_path)
        csv_length = len(im_paths)

        # dataset parameters
        self.root           = dataset_root
        self.loader         = self.pil_loader
        self.im_paths       = im_paths
        self.bbox           = bbox
        self.kp_loc         = kp_loc
        self.kp_cls         = kp_cls
        self.obj_cls        = obj_cls
        self.vp_labels      = vp_labels
        self.img_size       = im_size
        self.map_size       = map_size
        self.num_classes    = num_classes
        self.num_instances  = len(self.im_paths)
        self.transform      = transform

        # Print out dataset stats
        print("================================")
        print("Pascal3D (w/ Keypoints) Stats: ")
        print("CSV file length  : ", len(im_paths))
        print("Dataset size     : ", self.num_instances)
        print("Loading time (s) : ", time.time() - start_time)


    """
        __getitem__ method:
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
    """
    def __getitem__(self, index):

        # Load and transform image
        if self.root == None:
            c = self.im_paths[index]
        else:
            im_path = os.path.join(self.root, self.im_paths[index])

        bbox    = list(self.bbox[index])
        kp_loc  = list(self.kp_loc[index])
        kp_cls  = self.kp_cls[index]
        obj_cls = self.obj_cls[index]

        view    = self.vp_labels[index]

        # Transform labels
        azim, elev, tilt = (view + 360.) % 360.

        # Load and transform image
        img, kp_loc = self.loader(im_path, bbox, kp_loc)
        img = self.transform(img)

        # Generate keypoint map image, and kp class vector
        kpc_vec         = np.zeros( (34) )
        kpc_vec[kp_cls] = 1
        kp_class        = torch.from_numpy(kpc_vec).float()

        kpm_map         = self.generate_kp_map_chebyshev(kp_loc)
        kp_map          = torch.from_numpy(kpm_map).float()

        # construct unique key for statistics -- only need to generate imid and year
        _bb     = str(bbox[0]) + '-' + str(bbox[1]) + '-' + str(bbox[2]) + '-' + str(bbox[3])
        key_uid = self.im_paths[index] + '_'  + _bb + '_objc' + str(obj_cls) + '_kpc' + str(kp_cls)

        return img, azim, elev, tilt, obj_cls, kp_map, kp_class, key_uid

    """
        Retuns the Length of the dataset
    """
    def __len__(self):
        return self.num_instances

    """
        Image loader
        Inputs:
            path        absolute image path
            bbox        4-element tuple (x_min, y_min, x_max, y_max)
            flip        boolean for flipping image horizontally
            kp_loc      2-element tuple (x_loc, y_loc)
    """
    def pil_loader(self, path, bbox, kp_loc):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                # Calculate relative kp_loc position
                kp_loc[0] = float(kp_loc[0]-bbox[0])/float(bbox[2]-bbox[0])
                kp_loc[1] = float(kp_loc[1]-bbox[1])/float(bbox[3]-bbox[1])

                # Convert to RGB, crop, and resize
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                img = img.crop(box=bbox)
                img = img.resize( (self.img_size, self.img_size), Image.LANCZOS)

                return img, kp_loc

    """
        Convert CSV file to instances
    """
    def csv_to_instances(self, csv_path):
        # imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass
        # /z/.../datasets/pascal3d/Images/bus_pascal/2008_000032.jpg,5,117,488,273,9.186347,158.402214,1,4,1756,1799,1443

        df   = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 7, 8, 9, 12], axis=1)
        del(data_split[0])

        image_paths = np.squeeze(data_split[0]).tolist()

        # if self.root != None:
        #     image_paths = [path.split('pascal3d/')[1] for path in image_paths]

        bboxes      = data_split[1].tolist()
        kp_loc      = data_split[2].tolist()
        kp_class    = np.squeeze(data_split[3]).tolist()
        obj_class   = np.squeeze(data_split[4]).tolist()
        viewpoints  = np.array(data_split[5].tolist())

        return image_paths, bboxes, kp_loc, kp_class, obj_class, viewpoints


    """
        Generate Chbyshev-based map given a keypoint location
    """
    def generate_kp_map_chebyshev(self, kp):

        assert kp[0] >= 0. and kp[0] <= 1., kp
        assert kp[1] >= 0. and kp[1] <= 1., kp
        kp_map = np.ndarray( (self.map_size, self.map_size) )


        kp[0] = kp[0] * self.map_size
        kp[1] = kp[1] * self.map_size

        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                kp_map[i,j] = max( np.abs(i - kp[0]), np.abs(j - kp[1]))

        # Normalize by dividing by the maximum possible value, which is self.IMG_SIZE -1
        kp_map = kp_map / (1. * self.map_size)
        # kp_map = -2. * (kp_map - 0.5)

        return kp_map

