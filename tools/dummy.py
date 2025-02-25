#!/usr/bin/env python2

import os
import sys
import glob
import threading

import torch
import torchvision.io

from core.data.utils.vid import uniform_sample

import numpy as np
from PIL import Image
import math
import cv2

ls = ['S001C002P002R001A009',
      ]

from einops import rearrange
def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, frames, channel, height, width]
              -> [batch_size*frames, channel, height, width],
    batch_joints:[batch_size, frames, num_joints, 3]
              -> [batch_size*frames, num_joints, 3],
    batch_joints_vis:  [batch_size, frames, num_joints] torch.Size([4, 32, 25, 3])
                    -> [batch_size*frames, num_joints, 1],
    }
    '''
    batch_image = rearrange(batch_image,'b f c h w->(b f) c h w')
    batch_joints = rearrange(batch_joints, 'b f h w->(b f) h w')
    batch_joints_vis = rearrange(batch_joints_vis, 'b f h w->(b f) h w')
    np.zeros((20, 32, int(0.25 * 540), int(0.25 * 960), 3))
    P = np.zeros((20, 32, 25, 2))
    V = np.zeros((20, 32, 25, 1))
    batch_image = torch.tensor(batch_image).cuda().permute(0, 3, 1, 2)
    batch_joints_vis[:, :, :] = batch_joints_vis[:, :, 0:1]
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.shape[2] + padding)
    width = int(batch_image.shape[3] + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [255, 0, 0], 2)
            k = k + 1
    if os.path.isfile(file_name):
        os.remove(file_name)
    cv2.imwrite(file_name, ndarr)


def dummy():
    annot_path = '/mnt/data2/yyj/dataset/ntu_rgb/nturgb+d_numpy'
    images_path = '/mnt/data2/yyj/dataset/ntu_rgb/images'
    IM = np.zeros((20,32, int(0.25 * 540), int(0.25 * 960), 3))
    P = np.zeros((20,32, 25,2))
    V = np.zeros((20,32, 25,1))
    for j in range(1):
        sequence_id = 'S003C003P015R001A059'
        imp = os.path.join(images_path, sequence_id)
        images_ls = os.listdir(imp)
        images_ls.sort(key=lambda x: int(x[:-4]))
        annot_file = os.path.join(annot_path,
                                  sequence_id + '.npy')
        annot = np.load(annot_file)

        # frames = annot[1::2,0] # Divide video fps by 2
        frames = annot[:, 0]

        frame_idx = uniform_sample(len(frames), 32)

        frames = annot[frame_idx]  # frames[32,151]
        pose = frames[:, 1 + 3 * 25:]

        p = np.zeros((len(frames), 25, 2))
        v = np.ones((len(frames), 25, 1))
        p[:, :, 0] = pose[:, 0:25]
        p[:, :, 1] = pose[:, 25: 2 * 25]
        if np.isnan(p).any():
            print(sequence_id)
        p[np.isnan(p)] = 500
        p = p / (2 * 4)
        # print(p.shape)
        images = np.zeros((len(frames), int(0.25 * 540), int(0.25 * 960), 3))
        for i in range(32):
            image = cv2.imread(os.path.join(imp, images_ls[i]))  # 3,h,w
            image = cv2.resize(image, dsize=(int(0.25 * 960), int(0.25 * 540)))
            images[i] = image

        IM[j] = images
        P[j] = p
        V[j] = v

    file_name = os.path.join('/mnt/data2/yyj/test', sequence_id + '.jpg')
    save_batch_image_with_joints(IM, P, V, file_name)


if __name__ == "__main__":
    try:
        dummy()
    except Exception as e:
        print(e)
