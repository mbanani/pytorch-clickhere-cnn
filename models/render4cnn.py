import torch
import torch.nn as nn
from IPython import embed

class render4cnn(nn.Module):
    def __init__(self, weights_path = None):
        super(render4cnn, self).__init__()

        # define model
        self.conv4 = nn.Sequential(  
                        nn.Conv2d(3, 96, (11, 11), (4,4)),
                        nn.ReLU(),
                        nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True),
                        nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.),
                        nn.Conv2d(96, 256, (5, 5), (1,1), (2,2), 1,2),
                        nn.ReLU(),
                        nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True),
                        nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.),
                        nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
                        nn.ReLU(),
                        nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1),1,2),
                        nn.ReLU(),
                    )

        self.conv5 = nn.Sequential(  
                        nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1),1,2),
                        nn.ReLU(),
                        nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
                    )

        self.infer = nn.Sequential( 
                        nn.Linear(9216,4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(4096,4096),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                    )

        self.azim     = nn.Linear(4096, 12 * 360)
        self.elev     = nn.Linear(4096, 12 * 360)
        self.tilt     = nn.Linear(4096, 12 * 360)
        
        if weights_path is not None:
            self._initialize_weights(weights_path)        

    # weight initialization from torchvision/models/vgg.py
    def _initialize_weights(self, weights_path):
        state_dict = torch.load(weights_path)['model_state_dict']
        
        layers = [0, 4, 8, 10]
        for l in layers:
            self.conv4[l].weight.data.copy_( state_dict['conv4.'+str(l) + '.weight']) 
            self.conv4[l].bias.data.copy_(   state_dict['conv4.'+str(l) + '.bias']) 

        self.conv5[0].weight.data.copy_( state_dict['conv5.0.weight'])
        self.conv5[0].bias.data.copy_(   state_dict['conv5.0.bias']) 
        
        self.infer[0].weight.data.copy_(    state_dict['infer.0.weight'])
        self.infer[0].bias.data.copy_(      state_dict['infer.0.bias'])
        self.infer[3].weight.data.copy_(    state_dict['infer.3.weight'])
        self.infer[3].bias.data.copy_(      state_dict['infer.3.bias'])
    
        self.azim.weight.data.copy_(    state_dict['azim.0.weight'])
        self.azim.bias.data.copy_(      state_dict['azim.0.bias'])
        self.elev.weight.data.copy_(    state_dict['elev.0.weight'])
        self.elev.bias.data.copy_(      state_dict['elev.0.bias'])
        self.tilt.weight.data.copy_(    state_dict['tilt.0.weight'])
        self.tilt.bias.data.copy_(      state_dict['tilt.0.bias'])


    def forward(self, x, obj_class):
        # generate output 
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.infer(x)

        # mask on class
        azim = self.azim(x)
        azim = azim.view(-1, 12,360)
        azim = azim[torch.arange(x.shape[0]), obj_class, :]
        elev = self.elev(x)
        elev = elev.view(-1, 12,360)
        elev = elev[torch.arange(x.shape[0]), obj_class, :]
        tilt = self.tilt(x)
        tilt = tilt.view(-1, 12,360)
        tilt = tilt[torch.arange(x.shape[0]), obj_class, :]
        
        return azim, elev, tilt


