import os,sys, math
import torch
import numpy                    as np
import torchvision.transforms   as transforms
from datasets                   import pascal3d, pascal3d_kp

root_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = '/z/home/mbanani/datasets/pascal3d'


def get_data_loaders(dataset, batch_size, num_workers, model, num_classes = 12, flip = False, valid = 0.0):

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
                                num_classes = num_classes,
                                flip = True)

        test_set  = pascal3d_kp(csv_test,
                                dataset_root= dataset_root,
                                transform = test_transform,
                                im_size = image_size,
                                num_classes = num_classes)

    elif dataset == "pascalKP":
        csv_train = os.path.join(root_dir, 'data/pascal3d_kp_train.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_kp_valid.csv')

        train_set = pascal3d_kp(csv_train, dataset_root= dataset_root, transform = train_transform, im_size = image_size, flip = True)
        test_set  = pascal3d_kp(csv_test,  dataset_root= dataset_root, transform = test_transform,  im_size = image_size)
    else:
        print "Error in load_datasets: Dataset name not defined."


    # Generate validation dataset
    if valid > 0.0:
        valid_set   = train_set.generate_validation(valid)

    # Augment Training
    if flip:
        train_set.augment()
        print "Augmented Training Dataset - size : ", train_set.num_instances


    # Generate data loaders
    train_loader = torch.utils.data.DataLoader( dataset=train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                drop_last = True)

    test_loader  = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                drop_last = False)

    if valid > 0.0:
        print "Generated Validation Dataset - size : ", valid_set.num_instances
        valid_loader = torch.utils.data.DataLoader( dataset     = valid_set,
                                                    batch_size  = batch_size,
                                                    shuffle     = False,
                                                    pin_memory  = True,
                                                    num_workers = num_workers,
                                                    drop_last = False)
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader

# def get_data_path(machine, dataset):
#     if machine == 'focus':
#         if dataset == 'syn':
#             return get_data_path('z', dataset)
#         elif dataset == 'pascal':
#             return get_data_path('z', dataset)
#         else:
#             print "Error: Dataset argument not recognized. Set to either pascal or syn."
#             exit()
#     elif machine in ['lgn7', 'lgn6','lgn5', 'lgn4']:
#         if dataset == 'syn':
#             return '/data/mbanani/datasets/kp_render_synthetic/'
#         elif dataset == 'pascal':
#             return '/data/mbanani/datasets/pascal3d/'
#         else:
#             print "Error: Dataset argument not recognized. Set to either pascal or syn."
#             exit()
#     elif machine in ['lgn3', 'lgn2']:
#         if dataset == 'syn':
#             return '/scratch/mbanani/datasets/kp_render_synthetic/'
#         elif dataset == 'pascal':
#             return '/scratch/mbanani/datasets/pascal3d/'
#         else:
#             print "Error: Dataset argument not recognized. Set to either pascal or syn."
#             exit()
#     else:
#         if dataset == 'syn':
#             return '/z/home/mbanani/click-here-cnn/data/syn_images_cropped_bkg_overlaid'
#         elif dataset == 'pascal':
#             return '/z/home/mbanani/datasets/pascal3d'
#         else:
#             print "Error: Dataset argument not recognized. Set to either pascal or syn."
#             exit()
