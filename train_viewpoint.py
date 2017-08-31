import argparse
import os
import sys
import shutil
import time
import util

import numpy as np

import torch
from torch.autograd import Variable

from util           import evaluate_performance
from viewpoint_loss import ViewpointLoss
from datasets       import KP_PascalDataset
from models         import render4cnn, clickhere_cnn


def main(args):
    initialization_time = time.time()

    print "#############  Read in Database   ##############"
    data_loader, eval_data_loader = get_data_loaders(machine = args.machine,
                                                     dataset = args.dataset,
                                                     augment = args.flip,
                                                     batch_size = args.batch_size,
                                                     num_workers = args.num_workers)

    print "#############  Initiate Model     ##############"
    if args.model == 'render':
        model = render4cnn()
    elif args.model == 'clickhere':
        if util.render4cnn_weights == None:
            print "Error: Clickhere requires initialization with render4cnn weights. Set it in util.py."
            exit()
        model = clickhere_cnn(render4cnn(weights = 'lua', weights_path = util.render4cnn_weights))

    elif args.model == 'pretrained_render':
        if util.render4cnn_weights == None:
            print "Error: Weights path for pretrained render4cnn cannot be None. Set it in util.py."
            exit()
        model = render4cnn(weights = 'lua', weights_path = util.render4cnn_weights)

    elif args.model == 'pretrained_clickhere':
        if util.clickhere_weights == None:
            print "Error: Weights path for pretrained clickhere cannot be None. Set it in util.py."
            exit()
        model = clickhere_cnn(render4cnn(), weights_path = util.clickhere_weights)

    else:
        print "Error: unknown model choice. Exiting."
        exit()

    # Loss and Optimizer
    crit = ViewpointLoss()
    params = list(model.parameters())

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr = args.learning_rate, betas = (0.9, 0.999), eps=1e-8, weight_decay=0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum = 0.9, weight_decay = 0.0005)
    else:
        print "Error: Unknown choice for optimizer. Exiting."
        exit()

    # Train on GPU if available
    if torch.cuda.is_available():
        model.cuda()


    print "Time to initialize take: ", time.time() - initialization_time
    print "#############  Start Training     ##############"
    total_step = len(data_loader)

    for epoch in range(0, args.num_epochs):
        if epoch % args.eval_epoch == 0:
            model.eval()
            if 'pascal' in args.dataset:
                _, _ = eval_step(   model,
                                    data_loader,
                                    criterion = crit,
                                    message = "Evaluation(Train)",
                                    log_step = args.start_step + epoch * total_step,
                                    logger=curr_logger,
                                    datasplit = "train")


            curr_loss, curr_wacc = eval_step( model,
                                              eval_data_loader,
                                              criterion = crit,
                                              log_step = args.start_step + epoch * total_step,
                                              logger=curr_logger,
                                              message = "Evaluation(Valid)")

        if args.evaluate_only:
            exit()

        if epoch % args.save_epoch == 0 and epoch > 0:

            save_checkpoint( model = model,
                             optimizer = optimizer,
                             curr_epoch = epoch,
                             curr_step  = (total_step * epoch),
                             args = args,
                             curr_loss = curr_loss,
                             curr_wacc = curr_wacc,
                             filename =  ('model@epoch%d.pkl' %(epoch)))

        model.train()
        train_step( model,
                data_loader,
                crit, optimizer,
                epoch = epoch,
                step  = epoch * total_step)


def train_step(model, data_loader, criterion, optimizer, epoch, step):

    total_step = len(data_loader)
    epoch_time = time.time()
    batch_time = time.time()
    processing_time = 0
    loss_sum = 0.
    counter = 0

    for i, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(data_loader):
        counter = counter + 1
        training_time = time.time()

        # Set mini-batch dataset
        images = to_var(images, volatile=False)
        azim_label = to_var(azim_label)
        elev_label = to_var(elev_label)
        tilt_label = to_var(tilt_label)
        obj_class  = to_var(obj_class)

        if 'clickhere' in args.model :
            kp_map      = to_var(kp_map, volatile=False)
            kp_class    = to_var(kp_class, volatile=False)


        # Forward, Backward and Optimize
        model.zero_grad()

        if 'clickhere' in args.model :
            azim, elev, tilt = model(images, kp_map, kp_class)
        else:
            azim, elev, tilt = model(images)

        loss_a = criterion(azim, azim_label, obj_class)
        loss_e = criterion(elev, elev_label, obj_class)
        loss_t = criterion(tilt, tilt_label, obj_class)

        loss = loss_a + loss_e + loss_t

        loss.backward()
        optimizer.step()

        loss_sum += loss.data[0]

        processing_time += time.time() - training_time

        # Print log info
        if i % args.log_rate == 0 and i > 0:
            time_diff = time.time() - batch_time

            curr_batch_time = time_diff / (1.*args.log_rate)
            curr_train_per  = processing_time/time_diff
            curr_epoch_time = (time.time() - epoch_time) * (total_step / (i+1))
            curr_time_left  = (time.time() - epoch_time) * ((total_step - i) / (i+1))

            print "Epoch [%d/%d] Step [%d/%d]: Training Loss = %2.5f, Batch Time = %.2f sec, Time Left = %.1f mins." %( epoch, args.num_epochs,
                                                                                                                        i, total_step,
                                                                                                                        loss_sum / float(counter),
                                                                                                                        curr_batch_time,
                                                                                                                        curr_time_left / 60.)

            # Reset counters
            counter = 0
            loss_sum = 0.
            processing_time = 0
            batch_time = time.time()


def eval_step(model, data_loader, criterion = None, log_step = 0, logger = None, message="Evaluation", datasplit = 'val'):

    total_step = len(data_loader)
    epoch_error         = 0
    epoch_acc_thresh    = 0
    epoch_mean_correct  = 0
    epoch_loss_a        = 0.
    epoch_loss_e        = 0.
    epoch_loss_t        = 0.
    epoch_loss          = 0.

    epoch_type_correct = np.zeros(12, dtype=np.float16)
    epoch_type_total   = np.zeros(12, dtype=np.float16)

    epoch_geo_dists = [ [] for i in range(0, 12)]

    gt_vec = [ [] for x in range(0,12)]
    pr_vec = [ [] for x in range(0,12)]

    for step, (images, azim_label, elev_label, tilt_label, obj_class, kp_map, kp_class, key_uid) in enumerate(data_loader):

        images = to_var(images, volatile=True)
        azim_label = to_var(azim_label, volatile=True)
        elev_label = to_var(elev_label, volatile=True)
        tilt_label = to_var(tilt_label, volatile=True)

        if 'clickhere' in args.model :
            kp_map      = to_var(kp_map, volatile=True)
            kp_class    = to_var(kp_class, volatile=True)

        if 'clickhere' in args.model :
            azim, elev, tilt = model(images, kp_map, kp_class)
        else:
            azim, elev, tilt = model(images)

        if criterion:
            object_class  = to_var(obj_class)
            epoch_loss_a += criterion(azim, azim_label, object_class).data[0]
            epoch_loss_e += criterion(elev, elev_label, object_class).data[0]
            epoch_loss_t += criterion(tilt, tilt_label, object_class).data[0]


        error, acc_thresh, w_acc, type_correct,type_total, geo_dist, _ = evaluate_performance(  azim_label.data.cpu().numpy(),
                                                                                                elev_label.data.cpu().numpy(),
                                                                                                tilt_label.data.cpu().numpy(),
                                                                                                azim.data.cpu().numpy(),
                                                                                                elev.data.cpu().numpy(),
                                                                                                tilt.data.cpu().numpy(),
                                                                                                num_classes = 12,
                                                                                                obj_classes = obj_class.numpy(),
                                                                                                threshold = np.pi/6)


        epoch_geo_dists = [epoch_geo_dists[class_i] + geo_dist[class_i] for class_i in range(0, 12) ]

        altered_geo_dist = [dist_i for dist_i in geo_dist if dist_i != [] ]

        altered_geo_dist_median = [ np.median(type_dist) for type_dist in altered_geo_dist ]

        mean_median = np.mean(altered_geo_dist_median)

        epoch_error += error
        epoch_acc_thresh += acc_thresh
        epoch_type_correct += type_correct
        epoch_type_total += type_total

    altered_epoch_type_total = list(epoch_type_total)
    altered_epoch_type_correct = list(epoch_type_correct)

    for j in range(len(epoch_type_total)-1, -1, -1):
        if epoch_type_total[j] == 0:
            del altered_epoch_type_total[j]
            del altered_epoch_type_correct[j]

    altered_epoch_type_total = np.asarray(altered_epoch_type_total)
    altered_epoch_type_correct = np.asarray(altered_epoch_type_correct)

    altered_epoch_type_acc = altered_epoch_type_correct/altered_epoch_type_total

    altered_type_geo_dist = [dist_i for dist_i in epoch_geo_dists if dist_i != [] ]

    altered_geo_dist_median = [np.median(type_dist) * 180. / np.pi for type_dist in altered_type_geo_dist ]

    overall_mean_median = np.mean(altered_geo_dist_median)

    w_acc = np.mean(altered_epoch_type_acc)

    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "Median Error : ", int(1000 * overall_mean_median) / 1000., " degrees"
    print "Correct      : ", altered_epoch_type_correct
    print "Total num    : ", altered_epoch_type_total
    print "Type Acc     : ", [ int(10000 * a_type_acc) / 100. for a_type_acc in altered_epoch_type_acc ]
    print "Type Median  : ", [ int(1000 * a_type_med) / 1000. for a_type_med in altered_geo_dist_median ]
    print "Type Loss    : ", [epoch_loss_a/total_step, epoch_loss_e/total_step, epoch_loss_t/total_step]
    print "W. Accu      : ", int (10000 * w_acc) / 100., " %"
    print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    epoch_loss = float(epoch_loss)
    assert type(epoch_loss) == float, 'Error: Loss type is not float'
    return epoch_loss, w_acc


def save_checkpoint(model, optimizer, curr_epoch, curr_step, args, curr_loss, curr_wacc, filename):
    """
        Saves a checkpoint and updates the best loss and best weighted accuracy
    """
    is_best_loss = curr_loss < args.best_loss
    is_best_wacc = curr_wacc > args.best_wacc

    args.best_wacc = max(args.best_wacc, curr_wacc)
    args.best_loss = min(args.best_loss, curr_loss)

    state = {   'epoch':curr_epoch,
                'step': curr_step,
                'args': args,
                'state_dict': model.state_dict(),
                'val_loss': args.best_loss,
                'val_wacc': args.best_wacc,
                'optimizer' : optimizer.state_dict(),
             }

    path = os.path.join(args.experiment_path, filename)
    torch.save(state, path)
    if is_best_loss:
        shutil.copyfile(path, os.path.join(args.experiment_path, 'model_best_loss.pkl'))
    if is_best_wacc:
        shutil.copyfile(path, os.path.join(args.experiment_path, 'model_best_wacc.pkl'))

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_data_loaders(machine, dataset, augment, batch_size, num_workers):
    # Get dataset information
    if util.LMDB_data_path == None:
        print "Error: LMDB data dataset path is not set. Set it in util.py"
        exit()

    if dataset == "syn":
        dataset_root = os.path.join(util.LMDB_data_path, 'syn')
        train_set    = KP_PascalDataset(dataset_root, 'train', flip = augment)
        test_set     = KP_PascalDataset(dataset_root, 'test', flip = False)
    elif dataset == "pascal":
        dataset_root = os.path.join(util.LMDB_data_path, 'pacal')
        train_set = KP_PascalDataset(dataset_root, 'train', flip = augment)
        test_set  = KP_PascalDataset(dataset_root, 'test', flip = False)

    data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    eval_data_loader = torch.utils.data.DataLoader( dataset=test_set,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)

    return data_loader, eval_data_loader

if __name__ == '__main__':

    root_dir        = os.path.dirname(os.path.abspath(__file__))
    experiment_dir  = os.path.join(root_dir, 'experiments')

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_step',       type=int , default=10000)
    parser.add_argument('--save_epoch',      type=int , default=2)
    parser.add_argument('--eval_epoch',      type=int , default=5)
    parser.add_argument('--num_epochs',      type=int, default=100)
    parser.add_argument('--log_rate',        type=int, default=10)
    parser.add_argument('--batch_size',      type=int, default=128)
    parser.add_argument('--num_workers',     type=int, default=7)
    parser.add_argument('--learning_rate',   type=float, default=0.0001)
    parser.add_argument('--dataset',         type=str, default='syn')
    parser.add_argument('--model',           type=str, default='softmax')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--optimizer',       type=str,default='adam')

    args = parser.parse_args()

    args.experiment_path = os.path.join(experiment_dir, 'experiment_' + time.strftime("%m-%d_%H-%M-%S") + "_" + args.experiment_name)
    args.best_loss      = sys.float_info.max
    args.best_wacc      = 0.
    args.num_classes    = 12

    if args.experiment_name == None:
        args.experiment_name = ('%s_%s_%s_flip-%s'%(args.optimizer,
                                                    str(args.learning_rate),
                                                    args.model,
                                                    args.flip) )

    # Create model directory
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    print "Experiment path is : ", args.experiment_path
    print(args)
    main(args)
