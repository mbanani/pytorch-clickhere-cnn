from __future__ import division
import torch
import math

import numpy as np
import numbers
import types
import collections
import time
import os
import pandas

import numpy            as np
import torch.utils.data as data

from PIL            import Image
from ..util           import label_to_probs
from torchvision    import transforms

class KP_Dataset(data.Dataset):

    def __init__(self, dataset_root, datasplit, flip = False):

        start_time = time.time()

        self.num_classes = 12

        keys_file = open(dataset_root + '/' + datasplit + '/keys.txt')
        keys = keys_file.readlines()
        keys = [key.replace('\n', '') for key in keys]

        size_dataset = len(keys)

        if flip:
            self.keys = keys + keys
            self.flip = [False] * size_dataset + [True] * size_dataset
        else:
            self.keys = keys
            self.flip = [False] * size_dataset

        self.dataset_root = dataset_root + '/' + datasplit
        self.loader     = self.pil_loader

        print "csv file length: ", size_dataset

        # Normalization as instructed from pyTorch documentation
        self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(   mean=(0., 0., 0.),
                                                         std=(1./255., 1./255., 1./255.)
                                                     ),
                                transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                        std=(1., 1., 1.)
                                                    )
                                ])


        print "Dataset loaded in ", time.time() - start_time, " secs."
        print "Dataset size: ", len(self.keys)

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # Load and transform image
        key = self.keys[index]
        # _, _year, _imid, _, _objc, _bb, _ = key.split('_')
        key_splits = key.split('_')
        if len(key_splits) == 7:
            # Pascal Dataset -> unique_year_image_objID_objClass_BBOX_KP-info
            _, _year, _imid, _, _objc, _bb, _ = key_splits
        elif len(key_splits) == 11:
            # Pascal Dataset -> unique_model_image_azim_elev_tilt_dist_objID_objClass_BBOX_KP-info
            _, _year, _imid, _,_,_,_, _, _objc, _bb, _ = key_splits

        key_uid = _year + '_' + _imid + '_' + _objc + '_' + _bb



        im_path = self.dataset_root + '/image/' + self.keys[index] + '.png'
        vp_path = self.dataset_root + '/viewpoint_label/' + self.keys[index] + '.npy'
        kp_c_path = self.dataset_root + '/keypoint_class/' + self.keys[index] + '.npy'
        kp_m_path = self.dataset_root + '/chessboard_dt_map/' + self.keys[index] + '.npy'

        img = self.loader(im_path, self.flip[index])
        if self.transform is not None:
            img = self.transform(img)

        vp          = np.load(vp_path)
        kp_map      = np.load(kp_m_path) / 227.
        kp_class    = np.load(kp_c_path)
        vp_label    = tuple(vp[1:])
        obj_class   = vp[0]

        kp_class = torch.from_numpy(kp_class).float()
        kp_map   = torch.from_numpy(kp_map).float()

        # Transform labels
        azim, elev, tilt = label_to_probs( vp_label,
                                            obj_class,
                                            self.flip[index])

        # Load and transform label
        return img, azim, elev, tilt,obj_class, kp_map, kp_class, key_uid

    def __len__(self):
        return len(self.keys)

    def pil_loader(self, path, flip):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                # flip image
                if flip:
                    img.transpose(Image.FLIP_LEFT_RIGHT)

                return img
