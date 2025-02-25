import time

import os
import torchvision
import yaml
from alive_progress import alive_bar
from core.data.utils.vid import uniform_random_sample,uniform_sample

from core.config.cfg import cfg

TEST_MODE = 0
TRAIN_MODE = 1
EVAL_MODE = 2

class NTUImages(object):
    def __init__(self,dataset_path,eval_mode='cv',num_S=17, num_C=3, num_P=40, num_R=2, num_A=60) -> None:
        self.data_path = dataset_path
        end = time.time()
        print('-->loading ntu images')
        self.db = self._ntu_load_images(dataset_path,eval_mode,num_S, num_C, num_P, num_R, num_A)
        print('loaded ntu images, using time ', time.time()-end)

    def _ntu_load_images(self,dataset_path, eval_mode='cv',num_S=17, num_C=3, num_P=40, num_R=2, num_A=60):
        assert eval_mode in ['cs', 'cv'], \
            'Invalid evaluation mode {}'.format(eval_mode)

        cs_train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
                    25, 27, 28, 31, 34, 35, 38]
        cv_train = [2, 3]

        ntud_images_dir = os.path.join(dataset_path, 'images')
        images = [[], [], []]
        with alive_bar(len(range(num_S*num_C*num_P*num_R*num_A))) as bar:
            for s in range(1, num_S + 1):
                for c in range(1, num_C + 1):
                    for p in range(1, num_P + 1):
                        for r in range(1, num_R + 1):
                            for a in range(1, num_A + 1):
                                bar()
                                sequence_id = \
                                    'S%03dC%03dP%03dR%03dA%03d' % (s, c, p, r, a)

                                """均匀分段随机采样"""
                                dir_name = os.path.join(ntud_images_dir,
                                                        sequence_id)
                                if not os.path.isdir(dir_name):
                                    continue
                                frames = os.listdir(dir_name)
                                frames.sort(key=lambda x: int(x[:-4]))
                                if cfg['FULL_IMAGE']:
                                    frame_indexs = uniform_random_sample(len(frames), cfg['DATASET']['CLIP_SIZE'])
                                # else:
                                #     frame_indexs = get_fixed_clip_frames(len(frames), cfg['DATASET']['CLIP_SIZE'])
                                #

                                clipped_frames = []
                                for i in range(cfg['DATASET']['CLIP_SIZE']):
                                    """transform images"""
                                    if cfg['FULL_IMAGE']:
                                        image_file = os.path.join(dir_name,frames[frame_indexs[i]])
                                    else:
                                        image_file = os.path.join(dir_name,frames[i])
                                    if not os.path.exists(image_file):
                                        print(image_file, "doesn't exist")
                                        raise

                                    data_tensor = torchvision.io.read_image(image_file)
                                    clipped_frames.append(data_tensor)

                                if cfg['FULL_IMAGE']:
                                    clipped_frames.append(frame_indexs)
                                if eval_mode == 'cs':
                                    # TRAN_MODE = 1 TEST_MODE = 0
                                    mode = TRAIN_MODE if p in cs_train else TEST_MODE
                                else:
                                    mode = TRAIN_MODE if c in cv_train else TEST_MODE

                                images[mode].append(clipped_frames)

        return images
