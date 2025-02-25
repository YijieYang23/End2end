#!/usr/bin/env python2

import os
import shutil
import sys
from alive_progress import alive_bar

import numpy as np

def get_fixed_clip_frames(num_total_frames, num_clipped_frames):
    clips = []
    interval = num_total_frames / num_clipped_frames # >=1
    start = 0
    end = start + interval
    for i in range(num_clipped_frames):
        clips.append(int(start))
        start = end
        end = end + interval
    return clips


import cv2

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)


def copy_vid(input_path,subjects,
        num_C=3, num_P=40, num_R=2, num_A=60):

    if os.path.isdir(input_path) is False:
        raise Exception('Could not find the "{}" folder!'.format(input_path))
    num_C = 3
    num_P = 40
    num_R = 2
    num_A = 60
    num_S = len(subjects)
    with alive_bar(len(range(num_S * num_C * num_P*num_R*num_A))) as bar:
        for s in subjects:
            for c in range(1,num_C+1):
                for p in range(1,num_P+1):
                    for r in range(1,num_R+1):
                        for a in range(1,num_A+1):
                            bar()

                            sequence_id = 'S%03dC%03dP%03dR%03dA%03d' \
                                    % (s, c, p, r, a)

                            vid_path = os.path.join(input_path, sequence_id)

                            if os.path.isdir(vid_path) is False:
                                continue

                            frames = os.listdir(vid_path)

                            frames.sort(key=lambda x: int(x[:-4]))
                            frame_indexs = get_fixed_clip_frames(len(frames), num_clipped_frames=32)
                            safe_images = []
                            for i in range(32):
                                safe_images.append(frames[frame_indexs[i]])
                            if sequence_id == 'S001C001P001R001A001':
                                print(safe_images)
                            for i in range(len(frames)):
                                if frames[i] not in safe_images:
                                    os.remove(os.path.join(vid_path,frames[i]))

                            last = os.listdir(vid_path)
                            if len(last)!=32:
                                print(last)
                                print('erro when delete sequence_id')


if __name__ == "__main__":
    try:
        copy_vid(
            '/mnt/data2/yyj/dataset/ntu_rgb/images', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        # copy_vid(
        #         '/mnt/data2/yyj/dataset/ntu_rgb/images', [1])
    except Exception as e:
        print (e)

