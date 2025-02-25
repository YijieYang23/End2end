import numpy as np
import random
import logging
import cv2
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange

from core.data.utils.bbox import get_gt_bbox
from core.data.utils.bbox import bbox_to_objposwin
from core.data.utils.transform import get_affine_transform
from core.data.utils.transform import affine_transform
from core.data.utils.vid import uniform_random_sample
from core.data.utils.vid import uniform_sample

from core.data.ntuannot import NTUAnnot

logger = logging.getLogger(__name__)


class Feeder(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train, annot_dataset):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        self.is_train = is_train
        self.data_dir = cfg['DATA_DIR']
        self.ntu_image_size = cfg['DATASET']['dataset_image_size']
        self.read_image_size = cfg['DATASET']['read_image_size']
        self.image_size = np.array(cfg['MODEL']['IMAGE_SIZE'])
        self.heatmap_size = np.array(cfg['MODEL']['HEATMAP_SIZE'])
        self.ntu_scale_ratio = self.read_image_size[0] / self.ntu_image_size[0]
        self.clip_size = cfg['DATASET']['CLIP_SIZE']
        self.switch_interval = cfg['switch_interval'] if 'switch_interval' in cfg else 0
        self.annot_db = annot_dataset.db
        self.sampler = cfg['sampler'] if 'sampler' in cfg else 'uniform_sample'
        self.data_augmentation = cfg['DATASET']['DATA_AUGMENTATION']
        if self.data_augmentation:
            self.scale_range = cfg['DATASET']['scale_range']
            self.rotate_range = cfg['DATASET']['rotate_range']
        self.bbox_mode = cfg['DATASET']['bbox_mode']
        self.bbox_scale = cfg['DATASET']['bbox_scale']
        self.coord_representation = cfg['MODEL']['COORD_REPRESENTATION']
        if self.coord_representation == 'sa-simdr':
            self.simdr_split_ratio = cfg['MODEL']['SIMDR_SPLIT_RATIO']
        self.sigma = cfg['MODEL']['SIGMA']
        self.pixel_std = cfg['DATASET']['PIXEL_STD']
        self.num_joints = cfg['DATASET']['NUM_JOINTS']
        self.num_action = cfg['DATASET']['NUM_ACTION']

    def __len__(self, ):
        return len(self.annot_db['actions'][self.is_train])

    def __getitem__(self, idx):
        seq_id = self.annot_db['seq_ids'][self.is_train][idx]
        seq = self.annot_db['seqs'][self.is_train][idx]  # (2,103,17,2)
        frame_ids = self.annot_db['frame_ids_list'][self.is_train][idx]
        action = self.annot_db['actions'][self.is_train][idx]
        if len(frame_ids) > self.clip_size:
            if self.sampler == 'uniform_sample':
                clip_idxes = uniform_sample(len(frame_ids), self.clip_size)
            elif self.sampler == 'uniform_random_sample':
                clip_idxes = uniform_random_sample(len(frame_ids), self.clip_size)
            else:
                raise
            frame_ids = [frame_ids[i] for i in clip_idxes]
        elif len(frame_ids) < self.clip_size:
            raise ValueError(f'clip size greater than number of frames when processing {seq_id}')
        frame_idxes = [i - 1 for i in frame_ids]
        sampled_seq = seq[:, frame_idxes]
        num_people = len(sampled_seq)
        joints, joints_vis = self._get_joints_vis(sampled_seq[0])

        if self.switch_interval != 0:  # with person switching
            if num_people == 2:
                second_joints, second_joints_vis = self._get_joints_vis(sampled_seq[1])
                switch_idxes = []
                cout = self.switch_interval
                i = self.switch_interval
                while i < self.clip_size:
                    if cout > 0:
                        cout -= 1
                        if sum(second_joints_vis[i]) >= int(0.4 * self.clip_size):  # 如果第二个人的可见性太差，则还是用第一个人的
                            switch_idxes.append(i)
                    else:
                        i += self.switch_interval
                        cout = self.switch_interval
                        continue
                    i += 1
                joints[switch_idxes] = second_joints[switch_idxes]
                joints_vis[switch_idxes] = second_joints_vis[switch_idxes]

        """data augmentation"""
        if self.data_augmentation and self.is_train:
            rotation = random.uniform(self.rotate_range[0], self.rotate_range[1]) if random.random() <= 0.4 else 0.0
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1]) if random.random() <= 0.6 else 1.0
        else:
            scale_factor = 1.0
            rotation = 0.0

        """Compute the ground truth bounding box, if not given"""
        center = np.zeros((len(joints), 2), dtype=np.float32)
        if self.bbox_mode == '1-frame+':
            scale = np.zeros(2, dtype=np.float32)
        elif self.bbox_mode == '1-frame':
            scale = np.zeros((len(joints), 2), dtype=np.float32)
        else:
            raise ValueError('must choose a valid bbox mode')

        for i in range(self.clip_size):  # every frame
            bbox = get_gt_bbox(joints[i:i + 1], joints_vis[i:i + 1], self.read_image_size,
                               input_ratio=self.image_size[1] / self.image_size[0],
                               scale=self.bbox_scale, logkey=seq_id)
            center[i], temp_scale = bbox_to_objposwin(bbox)

            temp_scale = np.array([temp_scale[0] * 1.0 / self.pixel_std, temp_scale[1] * 1.0 / self.pixel_std],
                                  dtype=np.float32)
            if self.bbox_mode == '1-frame+':
                scale[0] = temp_scale[0] if temp_scale[0] > scale[0] else scale[0]
                scale[1] = temp_scale[1] if temp_scale[1] > scale[1] else scale[1]
            elif self.bbox_mode == '1-frame':
                scale[i, :] = temp_scale

        scale *= scale_factor

        """Set outsider body joints to invalid (-1e9)."""
        joints = rearrange(joints, 'frames joints dim->(frames joints) dim')
        joints[np.isnan(joints)] = -1e9
        joints = rearrange(joints, '(frames joints) dim->frames joints dim', frames=self.clip_size)

        joints_vis = np.expand_dims(joints_vis, axis=2)

        # 输入网络的32张图像,对于图像来说,输入都为高宽形式而非宽高形式(frames,channels=3,H,W)
        # <--> 与之对比的是对于pose标注,是先x后y(即先宽，后高),切记不要搞混
        images = torch.zeros((self.clip_size, 3, self.image_size[1], self.image_size[0]), dtype=torch.float32)
        targets = None  # (32,17,64,64) or (32,17,2,256*2)
        if self.coord_representation == 'sa-simdr':
            targets = np.zeros((self.clip_size, self.num_joints, int(self.image_size[0] * self.simdr_split_ratio) + int(
                self.image_size[1] * self.simdr_split_ratio)),
                               dtype=np.float32)
        elif self.coord_representation == 'heatmap':
            targets = np.zeros((self.clip_size, self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                               dtype=np.float32)

        # (32,25,1)
        target_weights = np.ones((self.clip_size, self.num_joints, 1), dtype=np.float32)

        for i in range(self.clip_size):
            """read images"""
            image_file = os.path.join(self.data_dir, 'images', seq_id,
                                      "%05d.jpg" % frame_ids[i])
            if not os.path.exists(image_file):
                logger.error(image_file, "doesn't exist")
                raise ValueError(image_file, "doesn't exist")

            data_tensor = torchvision.io.read_image(image_file)
            if data_tensor is None:
                logger.error('=> fail to read {}'.format(image_file))
                raise ValueError('Fail to read {}'.format(image_file))

            """transform images"""
            if self.bbox_mode == '1-frame+':
                trans = get_affine_transform(center[i], scale, rotation, np.array(self.image_size))
            elif self.bbox_mode == '1-frame':
                trans = get_affine_transform(center[i], scale[i], rotation, np.array(self.image_size))

            image = cv2.warpAffine(
                data_tensor.permute(1, 2, 0).numpy(),
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),  # 虽然输入是宽/高,但是输出的图片shape为(高,宽,3)
                flags=cv2.INTER_LINEAR)
            if self.transform:
                image = self.transform(image)  # (高,宽,3) -> (3,高,宽)
            images[i] = image

            """transform joints"""
            for j in range(self.num_joints):
                if joints_vis[i, j, 0] > 0.0:
                    joints[i, j, 0:2] = affine_transform(joints[i, j, 0:2], trans)

            """generate targets"""
            if self.coord_representation == 'sa-simdr':
                targets[i], target_weights[i] = self._generate_sa_simdr(joints[i], joints_vis[i])

            elif self.coord_representation == 'heatmap':
                targets[i], target_weights[i] = self._generate_target(joints[i], joints_vis[i])

        if not self.is_train:
            meta = {
                'joints': joints,  # (32,17,2)
                'joints_vis': joints_vis,  # (32,17,1)
                'center': center,  # (32,2)
                'scale': scale,  # '1-frame': (2); '1-frame+': (32,2)
                'rotation': rotation  # (1)
            }
        # print("using read_time",read_time)

        # print("get_item done, using time ",time.time()-end)

        """
        targets shape (32,25,64,64)   if heatmap represent
                      (32,25,2,256*2) if sa-simdr represent
        """

        if self.is_train:
            return images, torch.from_numpy(targets), torch.from_numpy(target_weights), torch.tensor([action])
        else:
            return images, torch.from_numpy(targets), torch.from_numpy(target_weights), torch.tensor([action]), meta

    def _get_joints_vis(self, joints):
        joints = joints.astype(np.float32)
        joints *= self.ntu_scale_ratio
        joints_vis = np.apply_along_axis(lambda x: 1 if x.all() else 0,
                                         axis=2, arr=(joints > 0))  # (frames,joints)
        joints[joints_vis == 0, :] = np.nan
        return joints, joints_vis

    def _adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= (self.image_size[0]) or ul[1] >= self.image_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight

    """generate sa-simdr representation"""

    def _generate_sa_simdr(self, pose, visible):
        '''
        :param pose:  [num_joints, 3]
        :param visible: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = visible[:, 0]

        target_x = np.zeros((self.num_joints,
                             int(self.image_size[0] * self.simdr_split_ratio)),
                            dtype=np.float32)
        target_y = np.zeros((self.num_joints,
                             int(self.image_size[1] * self.simdr_split_ratio)),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            target_weight[joint_id] = \
                self._adjust_target_weight(pose[joint_id], target_weight[joint_id], tmp_size)
            if target_weight[joint_id] == 0:
                continue

            mu_x = pose[joint_id][0] * self.simdr_split_ratio
            mu_y = pose[joint_id][1] * self.simdr_split_ratio

            x = np.arange(0, int(self.image_size[0] * self.simdr_split_ratio), 1, np.float32)
            y = np.arange(0, int(self.image_size[1] * self.simdr_split_ratio), 1, np.float32)

            v = target_weight[joint_id]
            if v > 0.5:
                target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * self.sigma ** 2))) / (
                        self.sigma * np.sqrt(np.pi * 2))
                target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * self.sigma ** 2))) / (
                        self.sigma * np.sqrt(np.pi * 2))

        target = np.concatenate([target_x, target_y], axis=1)

        return target, target_weight

    """generate heatmap"""

    def _generate_target(self, pose, visible):
        '''
        :param pose:  [num_joints, 3]
        :param visible: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = visible[:, 0]

        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(pose[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(pose[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)  # [size]
            y = x[:, np.newaxis]  # [size,1]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))  # [size]

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight


# S001C003P001R001A060 有问题
if __name__ == '__main__':
    from core.config.cfg import cfg
    from alive_progress import alive_bar

    ntu = NTUAnnot(cfg)

    train_dataset = Feeder(cfg, is_train=True, annot_dataset=ntu)

    info = train_dataset[1017]

    with alive_bar(len(train_dataset)) as bar:
        for i in range(len(train_dataset)):
            bar()
            info = train_dataset[i]
