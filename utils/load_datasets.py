
import torch
import torchvision.transforms   as transforms
import numpy                    as np
import os,sys, math
from datasets       import KP_Dataset, Pascal_Dataset, Synthetic_Dataset

def get_data_loaders(dataset, batch_size, num_workers, machine, flip = False):

    if dataset == "syn":
        csv_train = '/z/home/mbanani/click-here-cnn/data/image_keypoint_info/syn_train_image_keypoint_info.csv'
        csv_test  = '/z/home/mbanani/click-here-cnn/data/image_keypoint_info/syn_test_image_keypoint_info.csv'
        train_set = Synthetic_Dataset(csv_train, flip = flip)
        test_set  = Synthetic_Dataset(csv_test,  flip = False)

    elif dataset == "pascal":
        csv_train = '/z/home/mbanani/click-here-cnn/data/image_keypoint_info/pascal_train_image_keypoint_info.csv'
        csv_test  = '/z/home/mbanani/click-here-cnn/data/image_keypoint_info/pascal_test_image_keypoint_info.csv'
        train_set = Pascal_Dataset(csv_train, flip = flip)
        test_set  = Pascal_Dataset(csv_test,  flip = False)

    else:
        print "Error: Dataset argument not recognized. Set to either pascal or syn."
        exit()


    data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    eval_data_loader = torch.utils.data.DataLoader( dataset=test_set,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)

    return data_loader, eval_data_loader

def get_data_path(machine, dataset):
    if machine == 'focus':
        if dataset == 'syn':
            return get_data_path('z', dataset)
        elif dataset == 'pascal':
            return get_data_path('z', dataset)
        else:
            print "Error: Dataset argument not recognized. Set to either pascal or syn."
            exit()
    elif machine in ['lgn7', 'lgn6','lgn5', 'lgn4']:
        if dataset == 'syn':
            return '/data/mbanani/datasets/kp_render_synthetic/'
        elif dataset == 'pascal':
            return '/data/mbanani/datasets/pascal3d/'
        else:
            print "Error: Dataset argument not recognized. Set to either pascal or syn."
            exit()
    elif machine in ['lgn3', 'lgn2']:
        if dataset == 'syn':
            return '/scratch/mbanani/datasets/kp_render_synthetic/'
        elif dataset == 'pascal':
            return '/scratch/mbanani/datasets/pascal3d/'
        else:
            print "Error: Dataset argument not recognized. Set to either pascal or syn."
            exit()
    else:
        if dataset == 'syn':
            return '/z/home/mbanani/click-here-cnn/data/syn_images_cropped_bkg_overlaid'
        elif dataset == 'pascal':
            return '/z/home/mbanani/datasets/pascal3d'
        else:
            print "Error: Dataset argument not recognized. Set to either pascal or syn."
            exit()
