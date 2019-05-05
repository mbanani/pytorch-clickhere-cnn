import argparse, os, sys, shutil, time

import numpy as np
from IPython import embed

import torch

from util import SoftmaxVPLoss, Paths, get_data_loaders, kp_dict 
from models import clickhere_cnn, render4cnn
from util.torch_utils import save_checkpoint

def main(args):
    initialization_time = time.time()


    print("#############  Read in Database   ##############")
    train_loader, valid_loader = get_data_loaders(  dataset     = args.dataset,
                                                    batch_size  = args.batch_size,
                                                    num_workers = args.num_workers,
                                                    model       = args.model)

    print("#############  Initiate Model     ##############")
    if args.model == 'chcnn':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = clickhere_cnn(render4cnn(), weights_path = Paths.clickhere_weights)
        args.no_keypoint = False
    elif args.model == 'r4cnn':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = render4cnn(weights_path = Paths.render4cnn_weights)
        args.no_keypoint = True
    else:
        assert False, "Error: unknown model choice."

    # Loss functions
    criterion = SoftmaxVPLoss() 

    # Parameters to train
    if args.just_attention and (not args.no_keypoint):
        params = list(model.map_linear.parameters()) +list(model.cls_linear.parameters())
        params = params + list(model.kp_softmax.parameters()) +list(model.fusion.parameters())
        params = params + list(model.azim.parameters()) + list(model.elev.parameters())
        params = params + list(model.tilt.parameters())
    else:
        params = list(model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(params, lr = args.lr)
    
    # train/evaluate on GPU
    model.cuda()

    print("Time to initialize take: ", time.time() - initialization_time)
    print("#############  Start Training     ##############")
    total_step = len(train_loader)

    for epoch in range(0, args.num_epochs):

        if epoch % args.eval_epoch == 0:
            eval_step(  model       = model,
                        data_loader = valid_loader,
                        criterion   = criterion,
                        step        = epoch * total_step,
                        datasplit   = "valid")

        train_step( model        = model,
                    train_loader = train_loader,
                    criterion    = criterion,
                    optimizer    = optimizer,
                    epoch        = epoch,
                    step         = epoch * total_step)


def train_step(model, train_loader, criterion, optimizer, epoch, step):
    model.train()
    total_step      = len(train_loader)
    loss_sum        = 0.

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(train_loader):

        # Set mini-batch dataset
        images      = images.cuda()
        azim_label  = azim_label.cuda()
        elev_label  = elev_label.cuda()
        tilt_label  = tilt_label.cuda()
        obj_class   = obj_class.cuda()

        # Forward, Backward and Optimize
        model.zero_grad()

        if args.no_keypoint:
            azim, elev, tilt = model(images, obj_class)
        else:
            kp_map      = kp_map.cuda()
            kp_class    = kp_class.cuda()
            azim, elev, tilt = model(images, kp_map, kp_class, obj_class)

        loss_a = criterion(azim, azim_label)
        loss_e = criterion(elev, elev_label)
        loss_t = criterion(tilt, tilt_label)
        loss = loss_a + loss_e + loss_t

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        # Print log info
        if i % args.log_rate == 0 and i > 0:
            print("Epoch [%d/%d] Step [%d/%d]: Training Loss = %2.5f" %( epoch, args.num_epochs, i, total_step, loss_sum / (i + 1)))


def eval_step( model, data_loader,  criterion, step, datasplit):
    model.eval()

    total_step      = len(data_loader)
    epoch_loss_a    = 0.
    epoch_loss_e    = 0.
    epoch_loss_t    = 0.
    epoch_loss      = 0.
    results_dict    = kp_dict()

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(data_loader):

        if i % args.log_rate == 0:
            print("Evaluation of %s [%d/%d] " % (datasplit, i, total_step))

        # Set mini-batch dataset
        images      = images.cuda()
        azim_label  = azim_label.cuda()
        elev_label  = elev_label.cuda()
        tilt_label  = tilt_label.cuda()
        obj_class   = obj_class.cuda()

        if args.no_keypoint:
            azim, elev, tilt = model(images, obj_class)
        else:
            kp_map      = kp_map.cuda()
            kp_class    = kp_class.cuda()
            azim, elev, tilt = model(images, kp_map, kp_class, obj_class)

        # embed()
        epoch_loss_a += criterion(azim, azim_label).item()
        epoch_loss_e += criterion(elev, elev_label).item()
        epoch_loss_t += criterion(tilt, tilt_label).item()

        results_dict.update_dict( key_uid,
                            [azim.data.cpu().numpy(), elev.data.cpu().numpy(), tilt.data.cpu().numpy()],
                            [azim_label.data.cpu().numpy(), elev_label.data.cpu().numpy(), tilt_label.data.cpu().numpy()])


    type_accuracy, type_total, type_geo_dist = results_dict.metrics()

    geo_dist_median = [np.median(type_dist) * 180. / np.pi for type_dist in type_geo_dist if type_dist != [] ]
    type_accuracy   = [ type_accuracy[i] * 100. for i in range(0, len(type_accuracy)) if  type_total[i] > 0]
    w_acc           = np.mean(type_accuracy)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Type Acc_pi/6 : ", type_accuracy, " -> ", w_acc, " %")
    print("Type Median   : ", [ int(1000 * a_type_med) / 1000. for a_type_med in geo_dist_median ], " -> ", int(1000 * np.mean(geo_dist_median)) / 1000., " degrees")
    print("Type Loss     : ", [epoch_loss_a/total_step, epoch_loss_e/total_step, epoch_loss_t/total_step], " -> ", (epoch_loss_a + epoch_loss_e + epoch_loss_t ) / total_step)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--eval_epoch',      type=int , default=5)
    parser.add_argument('--log_rate',        type=int, default=10)
    parser.add_argument('--num_workers',     type=int, default=7)

    # training parameters
    parser.add_argument('--num_epochs',      type=int, default=100)
    parser.add_argument('--batch_size',      type=int, default=64)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--optimizer',       type=str,default='sgd')

    # experiment details
    parser.add_argument('--dataset',         type=str, default='pascal')
    parser.add_argument('--model',           type=str, default='pretrained_clickhere')
    parser.add_argument('--experiment_name', type=str, default= 'Test')
    parser.add_argument('--just_attention',  action="store_true",default=False)


    args = parser.parse_args()
    main(args)
