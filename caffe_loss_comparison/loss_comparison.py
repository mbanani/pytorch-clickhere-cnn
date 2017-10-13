import torch
import caffe
import numpy as np

from IPython import embed
from viewpoint_loss import ViewpointLoss

def calc_viewloss_vec(size, sigma):
    band    = np.linspace(-1*size, size, 1 + 2*size, dtype=np.int16)
    vec     = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
    prob    = np.exp(-1 * abs(vec) / sigma)
    prob    = prob / np.sum(prob)

    return band, prob

def label_to_probs(view_angles, object_class, num_classes = 12):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by 360 * object_class_id + angle
    :return:
    '''
    # Calculate object multiplier


    # Loss parameters taken directly from Render4CNN paper
    azim_band_width = 7     # 15 in paper
    azim_sigma = 5
    torch_label = np.zeros((len(view_angles), num_classes*360, 1 , 1), dtype=np.float)
    caffe_label = np.zeros((len(view_angles), num_classes*360, 1 , 1), dtype=np.float)

    # calculate probabilities
    azim_band, azim_prob = calc_viewloss_vec(azim_band_width, azim_sigma)

    for j in range(0, len(view_angles)):
        azim = view_angles[j]
        obj_mult = object_class[j]
        for i in azim_band:
            ind = np.mod(azim + i + 360, 360) + 360 * obj_mult
            torch_label[j, ind,0, 0] = azim_prob[i + azim_band_width]

        caffe_label[j, azim + 360*obj_mult,0, 0] = 1.

    return torch_label, caffe_label

data    = np.random.rand(10, 4320, 1, 1)
classes = [0,0,0,0,0,5,2,1,0,0]
Tlabels, Clabels  = label_to_probs([1,2,3,4,5,6,7,8,9,10],classes )


net = caffe.Net('test/loss.prototxt', caffe.TEST)

net.blobs['data'].data[...] = data
net.blobs['label'].data[...] = Clabels

caffe_loss = net.forward()

caffe_loss = float(caffe_loss['loss_azimuth'])

viewloss_mean = ViewpointLoss(mean=True)
viewloss_sum = ViewpointLoss(mean=False)

data_t  = torch.from_numpy(data).cuda()
label_t = torch.from_numpy(Tlabels).cuda()
objs_t  = torch.from_numpy(np.asarray( classes )).cuda()

data_t  = torch.autograd.Variable(data_t)
label_t = torch.autograd.Variable(label_t)
objs_t  = torch.autograd.Variable(objs_t)

data_t  = data_t.float()
label_t = label_t.float()
objs_t  = objs_t.float()


torch_lossM = viewloss_mean(data_t, label_t, objs_t)
torch_lossS = viewloss_sum(data_t, label_t, objs_t)


print "torch sum    :" , torch_lossS
print "torch mean   :" , torch_lossM
print "caffe        :" , caffe_loss
