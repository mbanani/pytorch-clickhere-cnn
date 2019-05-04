import torch
import torch.nn as nn
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
                        nn.Dropout(0.5),
                    )


        #Keypoint Stream
        kp_map      = nn.Linear(2116,2116)
        kp_class    = nn.Linear(34,34)
        kp_fuse     = nn.Linear(2150,169)

        # Fused layer
        self.fusion = nn.Sequential(
                        nn.Linear(4096 + 384, 4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                    )

        # Prediction layers
        azim        = nn.Linear(4096, 12 * 360)
        elev        = nn.Linear(4096, 12 * 360)
        tilt        = nn.Linear(4096, 12 * 360)


        self.pool_map    = nn.Sequential(nn.MaxPool2d( (5,5), (5,5), (1,1), ceil_mode=True))
        self.map_linear  = nn.Sequential( kp_map )
        self.cls_linear  = nn.Sequential( kp_class )
        self.kp_softmax  = nn.Sequential( kp_fuse, nn.Softmax() )

        self.azim = nn.Sequential(azim)
        self.elev = nn.Sequential(elev)
        self.tilt = nn.Sequential(tilt)

        self.init_weights(weights_path)


    def init_weights(self, weights_path):
        npy_dict = np.load(weights_path).item()

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

        kp_map.weight.data.copy_(state_dict['fc-keypoint-map']['weight'])
        kp_map.bias.data.copy_(state_dict['fc-keypoint-map']['bias'])
        kp_class.weight.data.copy_(state_dict['fc-keypoint-class']['weight'])
        kp_class.bias.data.copy_(state_dict['fc-keypoint-class']['bias'])
        kp_fuse.weight.data.copy_(state_dict['fc-keypoint-concat']['weight'])
        kp_fuse.bias.data.copy_(state_dict['fc-keypoint-concat']['bias'])

        azim.weight.data.copy_( state_dict['pred_azimuth'  ]['weight'] )
        elev.weight.data.copy_( state_dict['pred_elevation']['weight'] )
        tilt.weight.data.copy_( state_dict['pred_tilt'     ]['weight'] )

        azim.bias.data.copy_( state_dict['pred_azimuth'  ]['bias'] )
        elev.bias.data.copy_( state_dict['pred_elevation']['bias'] )
        tilt.bias.data.copy_( state_dict['pred_tilt'     ]['bias'] )

        self.infer[0].weight.data.normal_(0.0, 0.01)
        self.infer[0].bias.data.fill_(0)
        self.infer[3].weight.data.normal_(0.0, 0.01)
        self.infer[3].bias.data.fill_(0)

    def forward(self, images, kp_map, kp_cls):
        # Image Stream
        conv4 = self.conv4(images)
        im_stream = self.conv5(conv4)
        im_stream = im_stream.view(im_stream.size(0), -1)
        im_stream = self.infer(im_stream)

        # Keypoint Stream
        kp_map = kp_map.view(kp_map.size(0), -1)
        kp_map = self.map_linear(kp_map_flat)
        kp_cls = self.cls_linear(kp_cls)

        # Concatenate the two keypoint feature vectors
        kp_stream = torch.cat([features_map, features_cls], dim = 1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        kp_stream = self.kp_softmax(kp_stream)
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
        azim = self.azim(x)
        azim = azim.view(-1, 12, 360)
        azim = azim[torch.arange(x.shape[0]), obj_class, :]
        elev = self.elev(x)
        elev = elev.view(-1, 12, 360)
        elev = elev[torch.arange(x.shape[0]), obj_class, :]
        tilt = self.tilt(x)
        tilt = tilt.view(-1, 12, 360)
        tilt = tilt[torch.arange(x.shape[0]), obj_class, :]

        return azim, tilt, elev
