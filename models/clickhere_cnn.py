import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

class clickhere_cnn(nn.Module):
    def __init__(self, renderCNN, weights_path = None, num_classes = 12):
        super(clickhere_cnn, self).__init__()

        # Image Stream
        self.conv4 = renderCNN.conv4
        self.conv5 = renderCNN.conv5

        self.infer = nn.Sequential(
                        nn.Linear(9216,4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096,4096),
                        nn.ReLU(),
                        nn.Dropout(0.5))

        #Keypoint Stream
        self.kp_map = nn.Linear(2116,2116)
        self.kp_class = nn.Linear(34,34)
        self.kp_fuse = nn.Linear(2150,169)
        self.pool_map = nn.MaxPool2d( (5,5), (5,5), (1,1), ceil_mode=True)

        # Fused layer
        self.fusion = nn.Sequential(nn.Linear(4096 + 384, 4096), nn.ReLU(), nn.Dropout(0.5))

        # Prediction layers
        self.azim = nn.Linear(4096, 12 * 360)
        self.elev = nn.Linear(4096, 12 * 360)
        self.tilt = nn.Linear(4096, 12 * 360)

        if weights_path is not None:
            self.init_weights(weights_path)


    def init_weights(self, weights_path):
        npy_dict = np.load(weights_path, allow_pickle = True, encoding = 'latin1').item()

        state_dict = npy_dict
        # Convert parameters to torch tensors
        for key in list(npy_dict.keys()):
            state_dict[key]['weight'] = torch.from_numpy(npy_dict[key]['weight'])
            state_dict[key]['bias']   = torch.from_numpy(npy_dict[key]['bias'])

        self.conv4[0].weight.data.copy_(state_dict['conv1']['weight'])
        self.conv4[0].bias.data.copy_(state_dict['conv1']['bias'])
        self.conv4[4].weight.data.copy_(state_dict['conv2']['weight'])
        self.conv4[4].bias.data.copy_(state_dict['conv2']['bias'])
        self.conv4[8].weight.data.copy_(state_dict['conv3']['weight'])
        self.conv4[8].bias.data.copy_(state_dict['conv3']['bias'])
        self.conv4[10].weight.data.copy_(state_dict['conv4']['weight'])
        self.conv4[10].bias.data.copy_(state_dict['conv4']['bias'])
        self.conv5[0].weight.data.copy_(state_dict['conv5']['weight'])
        self.conv5[0].bias.data.copy_(state_dict['conv5']['bias'])

        self.infer[0].weight.data.copy_(state_dict['fc6']['weight'])
        self.infer[0].bias.data.copy_(state_dict['fc6']['bias'])
        self.infer[3].weight.data.copy_(state_dict['fc7']['weight'])
        self.infer[3].bias.data.copy_(state_dict['fc7']['bias'])
        self.fusion[0].weight.data.copy_(state_dict['fc8']['weight'])
        self.fusion[0].bias.data.copy_(state_dict['fc8']['bias'])

        self.kp_map.weight.data.copy_(state_dict['fc-keypoint-map']['weight'])
        self.kp_map.bias.data.copy_(state_dict['fc-keypoint-map']['bias'])
        self.kp_class.weight.data.copy_(state_dict['fc-keypoint-class']['weight'])
        self.kp_class.bias.data.copy_(state_dict['fc-keypoint-class']['bias'])
        self.kp_fuse.weight.data.copy_(state_dict['fc-keypoint-concat']['weight'])
        self.kp_fuse.bias.data.copy_(state_dict['fc-keypoint-concat']['bias'])

        self.azim.weight.data.copy_( state_dict['pred_azimuth'  ]['weight'] )
        self.elev.weight.data.copy_( state_dict['pred_elevation']['weight'] )
        self.tilt.weight.data.copy_( state_dict['pred_tilt'     ]['weight'] )

        self.azim.bias.data.copy_( state_dict['pred_azimuth'  ]['bias'] )
        self.elev.bias.data.copy_( state_dict['pred_elevation']['bias'] )
        self.tilt.bias.data.copy_( state_dict['pred_tilt'     ]['bias'] )


    def forward(self, images, kp_map, kp_cls, obj_class):
        # Image Stream
        conv4 = self.conv4(images)
        im_stream = self.conv5(conv4)
        im_stream = im_stream.view(im_stream.size(0), -1)
        im_stream = self.infer(im_stream)

        # Keypoint Stream
        kp_map = kp_map.view(kp_map.size(0), -1)
        kp_map = self.kp_map(kp_map)
        kp_cls = self.kp_class(kp_cls)

        # Concatenate the two keypoint feature vectors
        kp_stream = torch.cat([kp_map, kp_cls], dim = 1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        kp_stream = F.softmax(self.kp_fuse(kp_stream), dim=1)
        kp_stream = kp_stream.view(kp_stream.size(0) ,1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        kp_stream = kp_stream * conv4 # CHECK IF THIS DOES WHAT I THINK IT DOES!! TODO 
        kp_stream = kp_stream.sum(3).sum(2)

        # Concatenate fc7 and attended features
        fused_embed = torch.cat([im_stream, kp_stream], dim = 1)
        fused_embed = self.fusion(fused_embed)

        # Final inference
        azim = self.azim(fused_embed)
        elev = self.elev(fused_embed)
        tilt = self.tilt(fused_embed)

        # mask on class
        azim = self.azim(fused_embed)
        azim = azim.view(-1, 12, 360)
        azim = azim[torch.arange(fused_embed.shape[0]), obj_class, :]
        elev = self.elev(fused_embed)
        elev = elev.view(-1, 12, 360)
        elev = elev[torch.arange(fused_embed.shape[0]), obj_class, :]
        tilt = self.tilt(fused_embed)
        tilt = tilt.view(-1, 12, 360)
        tilt = tilt[torch.arange(fused_embed.shape[0]), obj_class, :]

        return azim, tilt, elev
