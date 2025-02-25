from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import yaml
from einops import rearrange
import numpy as np
import torchvision
import cv2
import os

from core.functions.inference import get_max_preds
'''mine'''
# arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]  # 23 <-> 11 <-> 10 ...
# rightHand = [11, 24]  # 11 <-> 24
# leftHand = [7, 22]  # 7 <-> 22
# legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]  # 19 <-> 18 <-> 17 ...
# body = [3, 2, 20, 1, 0]  # 3 <-> 2 <-> 20 ...

neighbor_link = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                             (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                             (1, 0), (3, 1), (2, 0), (4, 2)]

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, frames, channel, height, width]
              -> [batch_size*frames, channel, height, width],
    batch_joints:[batch_size, frames, num_joints, 3]
              -> [batch_size*frames, num_joints, 3],
    batch_joints_vis:  [batch_size, frames, num_joints] torch.Size([4, 32, 25, 3])
                    -> [batch_size*frames, num_joints, 1],
    }
    '''
    if len(batch_image.shape) == 5:
        batch_image = rearrange(batch_image,'batch_size frames channel height width -> (batch_size frames) channel height width')
    if len(batch_joints.shape) == 4:
        batch_joints = rearrange(batch_joints,'batch_size frames num_joints dim -> (batch_size frames) num_joints dim')
    if len(batch_joints_vis.shape) == 4:
        batch_joints_vis = rearrange(batch_joints_vis,'batch_size frames num_joints dim -> (batch_size frames) num_joints dim')
    batch_joints_vis[:,:,:] = batch_joints_vis[:,:,0:1]
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            joints_list = []
            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [255, 0, 0], 2)
                joints_list.append([joint_vis[0], int(joint[0]), int(joint[1])])
            for edge in neighbor_link:
                if joints_list[edge[0]][0] and joints_list[edge[1]][0]:
                    cv2.line(ndarr, (joints_list[edge[0]][1], joints_list[edge[0]][2]),
                             (joints_list[edge[1]][1], joints_list[edge[1]][2]),[0, 255, 0], 1)
            k = k + 1
    if os.path.isfile(file_name):
        os.remove(file_name)
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(debug,images,meta,joints_pred,prefix,print_batch_num=999):
    if not debug['DEBUG']:
        return

    batch_size = images.shape[0]
    idxes = np.arange(batch_size)

    if debug['SAVE_BATCH_IMAGES_GT']:
        if debug['MODE'] == 'all-frames':
            save_batch_image_with_joints(
                images[0:print_batch_num], meta['joints'][0:print_batch_num], meta['joints_vis'][0:print_batch_num],
                '{}_gt.jpg'.format(prefix)
            )
        else:
            save_batch_image_with_joints(
                images[:,[0,-1]], meta['joints'][:,[0,-1]], meta['joints_vis'][:,[0,-1]],
                '{}_gt.jpg'.format(prefix)
            )
    if debug['SAVE_BATCH_IMAGES_PRED']:
        if debug['MODE'] == 'all-frames':
            save_batch_image_with_joints(
                images[0:print_batch_num], joints_pred[0:print_batch_num], meta['joints_vis'][0:print_batch_num],
                '{}_pred.jpg'.format(prefix)
            )
        else:
            save_batch_image_with_joints(
                images[:,[0,-1]], joints_pred[:,[0,-1]], meta['joints_vis'][:,[0,-1]],
                '{}_pred.jpg'.format(prefix)
            )
    # if config.DEBUG.SAVE_HEATMAPS_GT:
    #     save_batch_heatmaps(
    #         input, target, '{}_hm_gt.jpg'.format(prefix)
    #     )
    # if config.DEBUG.SAVE_HEATMAPS_PRED:
    #     save_batch_heatmaps(
    #         input, output, '{}_hm_pred.jpg'.format(prefix)
    #     )
