# from __future__ import division
import torch
# import math

import numpy as np
# import numbers
# import types
# import collections
import time
# import os
import pandas

import numpy            as np
import torch.utils.data as data

from PIL            import Image
from util           import label_to_probs
from torchvision    import transforms

from IPython import embed

class Synthetic_Dataset(data.Dataset):


    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, csv_path, dataset_root = None, flip = False, num_classes = 12):

        start_time = time.time()

        # Load instance data from csv-file
        self.root           = dataset_root
        im_paths, bbox, kp_loc, kp_cls, obj_cls, vp_labels = self.csv_to_instances(csv_path)

        print "csv file length: ", len(im_paths)

        # assign data
        if flip:
            self.im_paths   = im_paths + im_paths
            self.bbox       = bbox + bbox
            self.kp_loc     = kp_loc + kp_loc
            self.kp_cls     = kp_cls + kp_cls
            self.obj_cls    = obj_cls + obj_cls
            self.vp_labels  = vp_labels + vp_labels
            self.flip       = [False] * len(im_paths) + [True] * len(im_paths)
        else:
            self.im_paths   = im_paths
            self.bbox       = bbox
            self.kp_loc     = kp_loc
            self.kp_cls     = kp_cls
            self.obj_cls    = obj_cls
            self.vp_labels  = vp_labels
            self.flip       = [False] * len(im_paths)

        # dataset parameters
        self.data_size      = len(self.im_paths)
        self.num_classes    = num_classes
        self.loader         = self.pil_loader

        # Normalization as for RenderForCNN and Clickhere CNN
        self.transform      = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(   mean=(0., 0., 0.),
                                                         std=(1./255., 1./255., 1./255.)
                                                     ),
                                transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                        std=(1., 1., 1.)
                                                    )
                                ])


        # Print out dataset stats
        print "Dataset loaded in ", time.time() - start_time, " secs."
        print "Dataset size: ", self.data_size

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
        kp_loc  = self.kp_loc[index]
        kp_cls  = self.kp_cls[index]
        obj_cls = self.obj_cls[index]
        view    = self.vp_labels[index]
        flip    = self.flip[index]

        # Transform labels
        azim, elev, tilt = label_to_probs(  view,
                                            obj_cls,
                                            flip)

        # Load and transform image
        img = self.loader(im_path, bbox = bbox, flip = flip)
        if self.transform is not None:
            img = self.transform(img)

        # construct keypoint map and keypoint class vector
        fake_kpc = np.zeros( (34) )
        # fake_kpc[kp_cls] = 1
        kp_class = torch.from_numpy(fake_kpc).float()
        fake_kpm = np.zeros( (227, 227) )
        kp_map   = torch.from_numpy(fake_kpm).float()

        #TODO construct unique key for statistics -- only need to generate imid and year
        # key_uid = _year + '_' + _imid + '_' + _objc + '_' + _bb
        _bb     = str(bbox[0]) + '-' + str(bbox[1]) + '-' + str(bbox[2]) + '-' + str(bbox[3])
        _imid   = '0000'
        key_uid = '0000' + '_' + _imid + '_' + str(obj_cls) + '_' + _bb

        # Load and transform label
        return img, azim, elev, tilt, obj_cls, kp_map, kp_class, key_uid

    def __len__(self):
        return self.data_size

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
    def pil_loader(self, path, bbox = None, rescale_size = 227 ,flip = False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # crop (TODO verify that it's the correct ordering!)
                if bbox != None:
                    img = img.crop(box=bbox)

                # verify that imresize uses LANCZOS
                img = img.resize( (rescale_size, rescale_size), Image.LANCZOS)

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                # Switch from H*W*C to C*H*W ?!

                # flip image
                if flip:
                    img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    def csv_to_instances(self, csv_path):
        # imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass
        # /z/.../datasets/pascal3d/Images/bus_pascal/2008_000032.jpg,5,117,488,273,9.186347,158.402214,1,4,1756,1799,1443

        df   = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 7, 8, 9, 12], axis=1)
        del(data_split[0])

        image_paths = np.squeeze(data_split[0]).tolist()
        if self.root != None:
            image_paths = [path.split('syn_images_cropped_bkg_overlaid/')[1] for path in image_paths]
        bboxes      = data_split[1].tolist()
        kp_loc      = data_split[2].tolist()
        kp_class    = np.squeeze(data_split[3]).tolist()
        obj_class   = np.squeeze(data_split[4]).tolist()
        viewpoints  = data_split[5].tolist()

        return image_paths, bboxes, kp_loc, kp_class, obj_class, viewpoints
