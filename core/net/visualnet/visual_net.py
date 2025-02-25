
import torch
import torch.nn.functional as F
from torch import nn

def relu_conv_bn(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                         nn.BatchNorm2d(out_channels))

# class MaxMinPooling(nn.Module):
#     def __init__(self,global_pooling = False):
#         super(MaxMinPooling, self).__init__()
#         if global_pooling:
#             self.max = nn.MaxPool2d (())
#             self.min = nn.MaxPool2d((2, 2), stride=(2, 2))
#         else:
#             self.max = nn.MaxPool2d((2,2),stride=(2,2))
#             self.min = nn.MaxPool2d((2, 2), stride=(2, 2))
#
#     def forward(self,x):
#         x1 = self.max(x)
#         x2 = self.min(x)
#         return x1-x2


class PredBlock(nn.Module):
    def __init__(self,in_channels,out_channels,last=False):
        super(PredBlock, self).__init__()
        self.last = last

        self.conv1 = relu_conv_bn(in_channels,int(in_channels/2),kernel_size=1)
        self.conv2 = relu_conv_bn(int(in_channels/2),in_channels,kernel_size=3,stride=1,padding=1)

        self.conv3 = relu_conv_bn(in_channels,in_channels,kernel_size=3,stride=1,padding=1)

        self.conv4 = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1))

        self.convback = relu_conv_bn(out_channels,in_channels,kernel_size=3,stride=1,padding=1)

        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        _x = self.conv3(x)
        x1 = F.max_pool2d(_x,(2,2))
        x2 = F.max_pool2d(-_x,(2,2))
        x = x1-x2
        action_hm = self.conv4(x)
        y = action_hm
        y1 = F.max_pool2d(y, y.shape[-2:])
        y2 = F.max_pool2d(-y, y.shape[-2:])
        y = y1-y2
        if not self.last:
            action_hm = self.upsample(action_hm)
            action_hm = self.convback(action_hm)
            x = residual + _x + action_hm
        return x,y.squeeze(-1).squeeze(-1)


class VisualBlock(nn.Module):
    # fuse (batch frames channels h w) -> (batch channels frames h*w)
    def __init__(self,cfg,num_features):  # num_features = 960
        super(VisualBlock, self).__init__()
        self.num_frames = cfg['DATASET']['CLIP_SIZE']
        self.num_actions = cfg['DATASET']['NUM_ACTION']
        self.conv = nn.Sequential(nn.Conv2d(num_features,256,kernel_size=1,stride=1,padding=0),
                                     nn.BatchNorm2d(256),
                                     nn.MaxPool2d(kernel_size=(2,2)))
        # self.pred_block1 = PredBlock(256, self.num_actions)
        # self.pred_block2 = PredBlock(256, self.num_actions)
        self.pred_block3 = PredBlock(256, self.num_actions)
        self.pred_block4 = PredBlock(256, self.num_actions,last=True)



    def forward(self,x):
        x = self.conv(x)
        # x, y1 = self.pred_block1(x)
        # x, y2 = self.pred_block2(x)
        x, y3 = self.pred_block3(x)
        _, y4 = self.pred_block4(x)

        return y4

if __name__ == "__main__":
## fuse (batch frames channels h w) -> (batch channels frames h*w)
    from core.config.cfg import cfg
    test = VisualBlock(cfg,960)
    fuse = torch.rand((6,960,32,4*4))
    pytorch_total_params = sum(p.numel() for p in test.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)
    res = test(fuse)
    print(res.shape)