from core.net.hrnet.hrnet import PoseHighResolutionNet as HRnet
from core.net.stgcn.st_gcn import Model as STGCN
from core.net.litehrnet.lite_hrnet import LiteHRNet
from core.net.visualnet.visual_net import VisualBlock

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, cfg, is_train=False):
        super(Model, self).__init__()
        self.is_train = is_train
        self.num_actions = cfg['DATASET']['NUM_ACTION']
        self.stgcn_detach = cfg['MODEL']['stgcn_detach'] if 'stgcn_detach' in cfg['MODEL'] else False
        self.multi_fuse = cfg['MODEL']['MULTI_FUSE']
        if cfg['BACKBONE'] == 'lite-hrnet':
            self.hrnet = LiteHRNet(cfg, in_channels=3)
        else:
            self.hrnet = HRnet(cfg)

        if cfg['MODEL']['COORD_REPRESENTATION'] == 'simdr' or cfg['MODEL']['COORD_REPRESENTATION'] == 'sa-simdr':
            stgcn_infeature = int(
                cfg['MODEL']['IMAGE_SIZE'][1] * cfg['MODEL']['SIMDR_SPLIT_RATIO'] + cfg['MODEL']['IMAGE_SIZE'][0] *
                cfg['MODEL']['SIMDR_SPLIT_RATIO'])
        elif cfg['MODEL']['COORD_REPRESENTATION'] == 'heatmap':
            stgcn_infeature = int(int(cfg['MODEL']['IMAGE_SIZE'][1] * cfg['MODEL']['IMAGE_SIZE'][0] / 16))
        else:
            raise

        self.stgcn = STGCN(stgcn_infeature, self.num_actions, {'layout': 'coco', 'strategy': 'spatial'},
                           cfg['MODEL']['EDGE_IMPORTANCE'], dropout=0.5)
        if self.multi_fuse:
            fuse_channel = cfg['MODEL']['EXTRA']['stages_spec']['num_channels'][-1][-1] \
                if cfg['BACKBONE'] == 'lite-hrnet' else cfg['MODEL']['EXTRA']['STAGE4']['NUM_CHANNELS'][-1]
            self.fuse_version = cfg['MODEL']['fuse_version'] if 'fuse_version' in cfg['MODEL'] else 1
            if self.fuse_version == 2:
                self.visualnet_detach = cfg['MODEL']['visualnet_detach'] if 'visualnet_detach' in cfg['MODEL'] else False
                self.v_block = VisualBlock(cfg, 3 * fuse_channel)
                # self.fcn = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
                #                          nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Linear(128, 60))
            if self.fuse_version == 1:
                self.t_cov = nn.Sequential(
                    nn.Conv1d(cfg['DATASET']['CLIP_SIZE'], 1, kernel_size=1),
                    nn.BatchNorm1d(1), nn.ReLU(inplace=True))
                self.fcn = nn.Conv2d(3 * fuse_channel, cfg['DATASET']['NUM_ACTION'], kernel_size=1)
            self.learnable_weights = nn.Parameter(torch.ones(self.num_actions))

    # input(32,25,2,256)
    # output(32,25,2,256)
    def forward(self, inputs):  # (2,32, 3, 256, 256)
        batch_size = inputs.shape[0]
        inputs = rearrange(inputs, 'b f c h w -> (b f) c h w')  # torch.Size([28, 32, 3, 144, 108])
        # if self.image_shuffle:
        #     # generate shuffle idx
        #     shuffle_idx = torch.randperm(inputs.shape[0])
        #     # generate unshuffle idx
        #     unshuffle_idx = torch.zeros_like(shuffle_idx)
        #     shuffle_tool = torch.arange(inputs.shape[0])
        #     unshuffle_idx[shuffle_idx] = shuffle_tool[shuffle_tool]
        #
        #     inputs = inputs[shuffle_idx]
        if self.multi_fuse:
            y1, fuse = self.hrnet(inputs)

            _y1 = rearrange(y1, '(batch frames) joints xyfeatures -> batch xyfeatures frames joints', batch=batch_size)
            if self.stgcn_detach:
                _y1 = _y1.detach()
            y2_1 = self.stgcn(_y1.unsqueeze(dim=4))

            if self.fuse_version == 1:
                height, width = fuse.shape[-2:]
                fuse = rearrange(fuse, '(batch frames) channels h w ->batch frames (channels h w)', batch=batch_size)
                fuse = self.t_cov(fuse).squeeze(1)
                fuse = rearrange(fuse, 'batch (channels h w) ->batch channels h w', h=height, w=width)
                fuse = F.avg_pool2d(fuse, fuse.size()[-2:])
                y2_2 = self.fcn(fuse).squeeze(-1).squeeze(-1)
                y2_3 = F.softmax(y2_1,dim=1).detach() + self.learnable_weights * F.softmax(y2_2,dim=1).detach()

            if self.fuse_version == 2:
                fuse = rearrange(fuse, '(batch frames) channels h w ->batch channels frames (h w)', batch=batch_size)
                if self.visualnet_detach:
                    fuse = fuse.detach()
                y2_2 = self.v_block(fuse)
                y2_3 = F.softmax(y2_1,dim=1).detach() + self.learnable_weights * F.softmax(y2_2,dim=1).detach()

            return y1, y2_1, y2_2, y2_3

        else:
            y1 = self.hrnet(inputs)
            _y1 = rearrange(y1, '(batch frames) joints xyfeatures -> batch xyfeatures frames joints', batch=batch_size)
            if self.stgcn_detach:
                _y1 = _y1.detach()
            y2 = self.stgcn(_y1.unsqueeze(dim=4))
            return y1, y2


if __name__ == "__main__":
    net = Model(is_train=True)
    x = torch.rand((2, 32, 3, 128, 128))

    y1, y2 = net(x)
    print(y1.shape)
    print(y2.shape)
