import os,sys, math
import torch
import numpy                    as np
import torchvision.transforms   as transforms
from datasets                   import pascal3d, pascal3d_kp

from IPython import embed
from .Paths import * 

root_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = pascal3d_root 


def get_data_loaders(dataset, batch_size, num_workers, model, num_classes = 12):

    image_size = 227
    train_transform   = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(   mean=(0., 0., 0.),
                                                 std=(1./255., 1./255., 1./255.)
                                             ),
                        transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                std=(1., 1., 1.)
                                            )
                        ])

    test_transform   = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(   mean=(0., 0., 0.),
                                                 std=(1./255., 1./255., 1./255.)
                                             ),
                        transforms.Normalize(   mean=(104, 116.668, 122.678),
                                                std=(1., 1., 1.)
                                            )
                        ])

    # # The New transform for ImageNet Stuff
    # new_transform   = transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                     std=(0.229, 0.224, 0.225))])


    if dataset == "pascal":
        csv_train = os.path.join(root_dir, 'data/pascal3d_train.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_valid.csv')

        train_set = pascal3d(csv_train, dataset_root= dataset_root, transform = train_transform, im_size = image_size)
        test_set  = pascal3d(csv_test,  dataset_root= dataset_root, transform = test_transform,  im_size = image_size)
    elif dataset == "pascalEasy":
        csv_train = os.path.join(root_dir, 'data/pascal3d_train_easy.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_valid_easy.csv')

        train_set = pascal3d(   csv_train, dataset_root= dataset_root,
                                transform = train_transform, im_size = image_size)
        test_set  = pascal3d(   csv_test,  dataset_root= dataset_root,
                                transform = test_transform,  im_size = image_size)

    elif dataset == "pascalVehKP":
        csv_train = os.path.join(root_dir, 'data/veh_pascal3d_kp_train.csv')
        csv_test  = os.path.join(root_dir, 'data/veh_pascal3d_kp_valid.csv')

        train_set = pascal3d_kp(csv_train,
                                dataset_root= dataset_root,
                                transform = train_transform,
                                im_size = image_size,
                                num_classes = num_classes)

        test_set  = pascal3d_kp(csv_test,
                                dataset_root= dataset_root,
                                transform = test_transform,
                                im_size = image_size,
                                num_classes = num_classes)

    elif dataset == "pascalKP":
        csv_train = os.path.join(root_dir, 'data/pascal3d_kp_train.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_kp_valid.csv')

        train_set = pascal3d_kp(csv_train, dataset_root= dataset_root, transform = train_transform, im_size = image_size)
        test_set  = pascal3d_kp(csv_test,  dataset_root= dataset_root, transform = test_transform,  im_size = image_size)
    else:
        print("Error in load_datasets: Dataset name not defined.")



    # Generate data loaders
    train_loader = torch.utils.data.DataLoader( dataset = train_set,
                                                batch_size =batch_size,
                                                shuffle = True ,
                                                num_workers =num_workers,
                                                drop_last   = True)

    test_loader  = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=batch_size,
                                                shuffle = False,
                                                num_workers=num_workers,
                                                drop_last = False)

    return train_loader, test_loader

