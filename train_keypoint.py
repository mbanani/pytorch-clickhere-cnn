import argparse
import os
import sys
import shutil
import time

import numpy as np
from IPython import embed

import torch

from util                       import ViewpointLoss, Logger, Paths
from util                       import get_data_loaders, vp_dict, kp_dict
from models                     import clickhere_cnn, render4cnn
from util.torch_utils           import to_var, save_checkpoint
from torch.optim.lr_scheduler   import MultiStepLR

def main(args):
    initialization_time = time.time()

    print "#############  Read in Database   ##############"
    train_loader, valid_loader, test_loader = get_data_loaders(   dataset     = args.dataset,
                                                    batch_size  = args.batch_size,
                                                    num_workers = args.num_workers,
                                                    model       = args.model,
                                                    flip        = args.flip,
                                                    num_classes = args.num_classes,
                                                    valid       = 0.0)

    print "#############  Initiate Model     ##############"
    if args.model == 'clickhere':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = clickhere_cnn(render4cnn(weights = 'lua', weights_path = Paths.render4cnn_weights), num_classes = args.num_classes)
        args.no_keypoint = False
    elif args.model == 'pretrained_clickhere':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = clickhere_cnn(render4cnn(), weights_path = Paths.clickhere_weights, num_classes = args.num_classes)
        args.no_keypoint = False
    elif args.model == 'pretrained_render':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = render4cnn(weights = 'lua', weights_path = Paths.render4cnn_weights, num_classes = args.num_classes)
        args.no_keypoint = True
    elif args.model == 'pretrained_FTrender':
        assert Paths.render4cnn_weights != None, "Error: Set render4cnn weights path in util/Paths.py."
        model = render4cnn(weights = 'npy', weights_path = Paths.ft_render4cnn_weights, num_classes = args.num_classes)
        args.no_keypoint = True
    else:
        assert False, "Error: unknown model choice."

    # Loss functions
    criterion = ViewpointLoss(num_classes = args.num_classes, weights = None) # train_loader.dataset.loss_weights)

    # Parameters to train
    if args.just_attention and (not args.no_keypoint):
        params = list(model.map_linear.parameters()) +list(model.cls_linear.parameters())
        params = params + list(model.kp_softmax.parameters()) +list(model.fusion.parameters())
        params = params + list(model.azim.parameters()) + list(model.elev.parameters())
        params = params + list(model.tilt.parameters())
    else:
        params = list(model.parameters())

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr = args.lr, betas = (0.9, 0.999), eps=1e-8, weight_decay=0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum = 0.9, weight_decay = 0.0005)
        scheduler = MultiStepLR( optimizer,
                                 milestones=range(0, args.num_epochs, 5),
                                 gamma=0.95)
    else:
        assert False, "Error: Unknown choice for optimizer."


    if args.resume is not None:
        print "Loading pretrained Module at %s " % (args.resume)
        checkpoint      = torch.load(args.resume)
        args.best_loss  = checkpoint['val_loss']
        args.best_acc   = checkpoint['val_acc']
        start_epoch     = checkpoint['epoch']
        start_step      = checkpoint['step']
        state_dict      = checkpoint['state_dict']

        print "Pretrained Model Val Accuracy is %f " % (args.best_acc)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # load params
        model.load_state_dict(state_dict)
    else:
        start_epoch     = 0
        start_step      = 0

    # if args.world_size > 1:
    #     print "Parallelizing Model"
    #     model = torch.nn.DataParallel(model, device_ids = range(0, args.world_size))

    # Train on GPU if available
    if torch.cuda.is_available():
        model.cuda()


    print "Time to initialize take: ", time.time() - initialization_time
    print "#############  Start Training     ##############"
    total_step = len(train_loader)

    for epoch in range(0, args.num_epochs):

        if epoch % args.eval_epoch == 0:
            if 'pascal' in args.dataset and args.evaluate_train:
                _, _ = eval_step(   model       = model,
                                    data_loader = train_loader,
                                    criterion   = criterion,
                                    step        = epoch * total_step,
                                    datasplit   = "train")


            curr_loss, curr_wacc = eval_step(   model       = model,
                                                data_loader = test_loader,
                                                criterion   = criterion,
                                                step        = epoch * total_step,
                                                datasplit   = "test")

            if valid_loader != None:
                curr_loss, curr_wacc = eval_step(   model       = model,
                                                    data_loader = valid_loader,
                                                    criterion   = criterion,
                                                    step        = epoch * total_step,
                                                    datasplit   = "valid")



        if args.evaluate_only:
            exit()
        #
        if epoch % args.save_epoch == 0 and epoch > 0:

            args = save_checkpoint(  model      = model,
                                     optimizer  = optimizer,
                                     curr_epoch = epoch,
                                     curr_step  = (total_step * epoch),
                                     args       = args,
                                     curr_loss  = curr_loss,
                                     curr_acc   = curr_wacc,
                                     filename   = ('model@epoch%d.pkl' %(epoch)))

        if args.optimizer == 'sgd':
            scheduler.step()

        logger.add_scalar_value("Misc/Epoch Number", epoch, step=epoch * total_step)
        train_step( model        = model,
                    train_loader = train_loader,
                    criterion    = criterion,
                    optimizer    = optimizer,
                    epoch        = epoch,
                    step         = epoch * total_step,
                    valid_loader = valid_loader,
                    valid_type   = "valid")

    # Final save of the model
    args = save_checkpoint(  model      = model,
                             optimizer  = optimizer,
                             curr_epoch = epoch,
                             curr_step  = (total_step * epoch),
                             args       = args,
                             curr_loss  = curr_loss,
                             curr_acc   = curr_wacc,
                             filename   = ('model@epoch%d.pkl' %(epoch)))

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None, valid_type = "valid"):
    model.train()
    total_step      = len(train_loader)
    epoch_time      = time.time()
    batch_time      = time.time()
    processing_time = 0
    loss_sum        = 0.
    counter         = 0

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(train_loader):
        counter = counter + 1
        training_time = time.time()

        # Set mini-batch dataset
        images      = to_var(images, volatile=False)
        azim_label  = to_var(azim_label)
        elev_label  = to_var(elev_label)
        tilt_label  = to_var(tilt_label)
        obj_class   = to_var(obj_class)

        if (not args.no_keypoint):
            kp_map      = to_var(kp_map, volatile=False)
            kp_class    = to_var(kp_class, volatile=False)

        # Forward, Backward and Optimize
        model.zero_grad()

        if args.no_keypoint:
            azim, elev, tilt = model(images)
        else:
            azim, elev, tilt = model(images, kp_map, kp_class)

        loss_a = criterion(azim, azim_label, obj_class)
        loss_e = criterion(elev, elev_label, obj_class)
        loss_t = criterion(tilt, tilt_label, obj_class)

        loss = loss_a + loss_e + loss_t

        loss_sum += loss.data[0]

        loss.backward()
        optimizer.step()

        # Log losses
        logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/train_azim",  loss_a.data[0] , step=step + i)
        logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/train_elev",  loss_e.data[0] , step=step + i)
        logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/train_tilt",  loss_t.data[0] , step=step + i)
        logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/train_sum", loss.data[0] , step=step + i)

        processing_time += time.time() - training_time

        # Print log info
        if i % args.log_rate == 0 and i > 0:
            time_diff = time.time() - batch_time

            curr_batch_time = time_diff / (1.*args.log_rate)
            curr_train_per  = processing_time/time_diff
            curr_epoch_time = (time.time() - epoch_time) * (total_step / (i+1.))
            curr_time_left  = (time.time() - epoch_time) * ((total_step - i) / (i+1.))

            print "Epoch [%d/%d] Step [%d/%d]: Training Loss = %2.5f, Batch Time = %.2f sec, Time Left = %.1f mins." %( epoch, args.num_epochs,
                                                                                                                        i, total_step,
                                                                                                                        loss_sum / float(counter),
                                                                                                                        curr_batch_time,
                                                                                                                        curr_time_left / 60.)

            logger.add_scalar_value("Misc/batch time (s)",    curr_batch_time,        step=step + i)
            logger.add_scalar_value("Misc/Train_%",           curr_train_per,         step=step + i)
            logger.add_scalar_value("Misc/epoch time (min)",  curr_epoch_time / 60.,  step=step + i)
            logger.add_scalar_value("Misc/time left (min)",   curr_time_left / 60.,   step=step + i)

            # Reset counters
            counter = 0
            loss_sum = 0.
            processing_time = 0
            batch_time = time.time()

        if valid_loader != None and i % args.eval_step == 0 and i > 0:
            model.eval()
            _, _ = eval_step(   model       = model,
                                data_loader = valid_loader,
                                criterion   = criterion,
                                step        = epoch * total_step,
                                datasplit   = valid_type)

            model.train()


def eval_step( model, data_loader,  criterion, step, datasplit):
    model.eval()
    total_step      = len(data_loader)
    start_time      = time.time()
    epoch_loss_a    = 0.
    epoch_loss_e    = 0.
    epoch_loss_t    = 0.
    epoch_loss      = 0.
    results_dict    = vp_dict()

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(data_loader):

        if i % args.log_rate == 0:
            print "Evaluation of %s [%d/%d] Time Elapsed: %f " % (datasplit, i, total_step, time.time() - start_time)

        images = to_var(images, volatile=True)
        azim_label = to_var(azim_label, volatile=True)
        elev_label = to_var(elev_label, volatile=True)
        tilt_label = to_var(tilt_label, volatile=True)


        if args.no_keypoint:
            azim, elev, tilt = model(images)
        else:
            kp_map      = to_var(kp_map, volatile=False)
            kp_class    = to_var(kp_class, volatile=False)
            azim, elev, tilt = model(images, kp_map, kp_class)

        # embed()
        object_class  = to_var(obj_class)
        epoch_loss_a += criterion(azim, azim_label, object_class).data[0]
        epoch_loss_e += criterion(elev, elev_label, object_class).data[0]
        epoch_loss_t += criterion(tilt, tilt_label, object_class).data[0]

        results_dict.update_dict( object_class.data.cpu().numpy(),
                            [azim.data.cpu().numpy(), elev.data.cpu().numpy(), tilt.data.cpu().numpy()],
                            [azim_label.data.cpu().numpy(), elev_label.data.cpu().numpy(), tilt_label.data.cpu().numpy()])


    type_accuracy, type_total, type_geo_dist = results_dict.metrics()

    geo_dist_median = [np.median(type_dist) * 180. / np.pi for type_dist in type_geo_dist if type_dist != [] ]
    type_accuracy   = [ type_accuracy[i] * 100. for i in range(0, len(type_accuracy)) if  type_total[i] > 0]
    w_acc           = np.mean(type_accuracy)

    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "Type Acc_pi/6 : ", type_accuracy, " -> ", w_acc, " %"
    print "Type Median   : ", [ int(1000 * a_type_med) / 1000. for a_type_med in geo_dist_median ], " -> ", int(1000 * np.mean(geo_dist_median)) / 1000., " degrees"
    print "Type Loss     : ", [epoch_loss_a/total_step, epoch_loss_e/total_step, epoch_loss_t/total_step], " -> ", (epoch_loss_a + epoch_loss_e + epoch_loss_t ) / total_step
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    logger.add_scalar_value("(" + args.dataset + ") Median Geodsic Error/" + datasplit + "_mean",   np.mean(geo_dist_median),step=step)
    logger.add_scalar_value("(" + args.dataset + ") Median Geodsic Error/" + datasplit + "_bus",    geo_dist_median[0],step=step)
    logger.add_scalar_value("(" + args.dataset + ") Median Geodsic Error/" + datasplit + "_car",    geo_dist_median[1],step=step)
    logger.add_scalar_value("(" + args.dataset + ") Median Geodsic Error/" + datasplit + "_mbike",  geo_dist_median[2],step=step)

    logger.add_scalar_value("(" + args.dataset + ") Accuracy_30deg/" + datasplit + "_mean",     np.mean(type_accuracy),step=step)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy_30deg/" + datasplit + "_bus",      type_accuracy[0],step=step)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy_30deg/" + datasplit + "_car",      type_accuracy[1],step=step)
    logger.add_scalar_value("(" + args.dataset + ") Accuracy_30deg/" + datasplit + "_mbike",    type_accuracy[2],step=step)

    logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/" + datasplit +"_azim",  epoch_loss_a / total_step, step=step)
    logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/" + datasplit +"_elev",  epoch_loss_e / total_step, step=step)
    logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/" + datasplit +"_tilt",  epoch_loss_t / total_step, step=step)
    logger.add_scalar_value("(" + args.dataset + ") Viewpoint Loss/" + datasplit +"_sum",   (epoch_loss_a + epoch_loss_e + epoch_loss_t ) / total_step, step=step)

    epoch_loss = float(epoch_loss)
    assert type(epoch_loss) == float, 'Error: Loss type is not float'
    return epoch_loss, w_acc



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--save_epoch',      type=int , default=10)
    parser.add_argument('--eval_epoch',      type=int , default=1)
    parser.add_argument('--eval_step',       type=int , default=1000)
    parser.add_argument('--log_rate',        type=int, default=10)
    parser.add_argument('--num_workers',     type=int, default=7)

    # training parameters
    parser.add_argument('--num_epochs',      type=int, default=100)
    parser.add_argument('--batch_size',      type=int, default=256)
    parser.add_argument('--lr',              type=float, default=0.01)
    parser.add_argument('--optimizer',       type=str,default='sgd')

    # experiment details
    parser.add_argument('--dataset',         type=str, default='pascalKP')
    parser.add_argument('--model',           type=str, default='pretrained_clickhere')
    parser.add_argument('--experiment_name', type=str, default= 'Test')
    parser.add_argument('--evaluate_only',   action="store_true",default=False)
    parser.add_argument('--evaluate_train',  action="store_true",default=False)
    parser.add_argument('--flip',            action="store_true",default=False)
    parser.add_argument('--just_attention',  action="store_true",default=False)
    parser.add_argument('--num_classes',     type=int, default=12)
    parser.add_argument('--resume',           type=str, default=None)
    # parser.add_argument('--world_size',      type=int, default=1)


    args = parser.parse_args()


    root_dir                    = os.path.dirname(os.path.abspath(__file__))
    experiment_result_dir       = os.path.join(root_dir, os.path.join('experiments',args.dataset))
    args.full_experiment_name   = ("exp_%s_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), args.dataset, args.experiment_name) )
    args.experiment_path        = os.path.join(experiment_result_dir, args.full_experiment_name)
    args.best_loss              = sys.float_info.max
    args.best_acc               = 0.

    # Create model directory
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    print "Experiment path is : ", args.experiment_path
    print(args)

    # Define Logger
    log_name    = args.full_experiment_name
    logger      = Logger(os.path.join(Paths.tensorboard_logdir, log_name))

    main(args)
