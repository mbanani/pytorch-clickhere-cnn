import torch
import numpy as np
import time
import pandas
import os

import numpy            as np
import torch.utils.data as data

from PIL            import Image
from .vp_util        import label_to_probs
from torchvision    import transforms
import copy
import random

from IPython import embed

class pascal3d(data.Dataset):
    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, csv_path, dataset_root = None, im_size = 227, transform = None, just_easy = False, num_classes = 12):

        start_time = time.time()

        # Load instance data from csv-file
        im_paths, bbox, obj_cls, vp_labels = self.csv_to_instances(csv_path)
        print("csv file length: ", len(im_paths))

        # dataset parameters
        self.root           = dataset_root
        self.loader         = self.pil_loader
        self.im_paths   = im_paths
        self.bbox       = bbox
        self.obj_cls    = obj_cls
        self.vp_labels  = vp_labels
        self.flip       = [False] * len(im_paths)

        self.im_size        = im_size
        self.num_classes    = num_classes
        self.num_instances  = len(self.im_paths)
        assert transform   != None
        self.transform      = transform

        # Set weights for loss
        class_hist          = np.histogram(obj_cls, list(range(0, self.num_classes+1)))[0]
        mean_class_size     = np.mean(class_hist)
        self.loss_weights   = mean_class_size / class_hist

        # Print out dataset stats
        print("Dataset loaded in ", time.time() - start_time, " secs.")
        print("Dataset size: ", self.num_instances)

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # Load and transform image
        if self.root == None:
            im_path = self.im_paths[index]
        else:
            im_path = os.path.join(self.root, self.im_paths[index])

        bbox    = self.bbox[index]
        obj_cls = self.obj_cls[index]
        view    = self.vp_labels[index]
        flip    = self.flip[index]

        # Transform labels
        azim, elev, tilt = (view + 360.) % 360.

        # Load and transform image
        img = self.loader(im_path, bbox = bbox, flip = flip)
        if self.transform is not None:
            img = self.transform(img)


        # construct unique key for statistics -- only need to generate imid and year
        _bb     = str(bbox[0]) + '-' + str(bbox[1]) + '-' + str(bbox[2]) + '-' + str(bbox[3])
        key_uid = self.im_paths[index] + '_'  + _bb + '_objc' + str(obj_cls) + '_kpc' + str(0)

        return img, azim, elev, tilt, obj_cls, -1, -1, key_uid

    def __len__(self):
        return self.num_instances

    """
        Loads images and applies the following transformations
            1. convert all images to RGB
            2. crop images using bbox (if provided)
            3. resize using LANCZOS to rescale_size
            4. convert from RGB to BGR
            5. (? not done now) convert from HWC to CHW
            6. (optional) flip image

        TODO: once this works, convert to a relative path, which will matter for
              synthetic data dataset class size.
    """
    def pil_loader(self, path, bbox = None ,flip = False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                img = img.crop(box=bbox)

                # verify that imresize uses LANCZOS
                img = img.resize( (self.im_size, self.im_size), Image.LANCZOS)

                # flip image
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    def csv_to_instances(self, csv_path):
        df   = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 6, 9], axis=1)
        del(data_split[0])

        image_paths = np.squeeze(data_split[0]).tolist()
        bboxes      = data_split[1].tolist()
        obj_class   = np.squeeze(data_split[2]).tolist()
        viewpoints  = np.array(data_split[3].tolist())

        return image_paths, bboxes, obj_class, viewpoints

    def augment(self):
        self.im_paths   = self.im_paths  + self.im_paths
        self.bbox       = self.bbox      + self.bbox
        self.obj_cls    = self.obj_cls   + self.obj_cls
        self.vp_labels  = self.vp_labels + self.vp_labels
        self.flip       = self.flip      + [True] * self.num_instances
        assert len(self.flip) == len(self.im_paths)
        self.num_instances = len(self.im_paths)
        print("Augmented dataset. New size: ", self.num_instances)

    def generate_validation(self, ratio = 0.1):
        assert ratio > (2.*self.num_classes/float(self.num_instances)) and ratio < 0.5

        random.seed(a = 2741998)

        valid_class     = copy.deepcopy(self)

        valid_size      = int(ratio * self.num_instances)
        train_size      = self.num_instances - valid_size
        train_instances = list(range(0, self.num_instances))
        valid_instances = random.sample(train_instances, valid_size)
        train_instances = [x for x in train_instances if x not in valid_instances]

        assert train_size == len(train_instances) and valid_size == len(valid_instances)

        valid_class.im_paths        = [ self.im_paths[i]     for i in sorted(valid_instances) ]
        valid_class.bbox            = [ self.bbox[i]            for i in sorted(valid_instances) ]
        valid_class.obj_cls         = [ self.obj_cls[i]         for i in sorted(valid_instances) ]
        valid_class.vp_labels       = [ self.vp_labels[i]       for i in sorted(valid_instances) ]
        valid_class.flip            = [ self.flip[i]           for i in sorted(valid_instances) ]
        valid_class.num_instances   = valid_size

        self.im_paths            = [ self.im_paths[i]     for i in sorted(train_instances) ]
        self.bbox                = [ self.bbox[i]            for i in sorted(train_instances) ]
        self.obj_cls             = [ self.obj_cls[i]         for i in sorted(train_instances) ]
        self.vp_labels           = [ self.vp_labels[i]       for i in sorted(train_instances) ]
        self.flip                 = [ self.flip[i]           for i in sorted(train_instances) ]
        self.num_instances       = train_size

        return valid_class
