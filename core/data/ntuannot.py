import time
import pickle
import random
import os
from alive_progress import alive_bar

TEST_MODE = 0
TRAIN_MODE = 1
EVAL_MODE = 2


class NTUAnnot(object):
    def __init__(self, cfg) -> None:
        self.random_pick_half = cfg['half_dataset'] if 'half_dataset' in cfg else False
        self.data_dir = cfg['DATA_DIR']
        self.eval_mode = cfg['TRAIN']['EVAL_MODE']
        self.num_S = cfg['NUM_S'] if not cfg['PURE_VALID'] else 1
        self.clip_size = cfg['DATASET']['CLIP_SIZE']
        end = time.time()
        print('-->loading ntu annot')
        seq_ids, seqs, frame_ids_list, actions = self._ntu_load_annotations()
        print('loaded ntu annot, using time ', time.time() - end)
        self.db = {}
        self.db['seq_ids'] = seq_ids
        self.db['seqs'] = seqs
        self.db['frame_ids_list'] = frame_ids_list
        self.db['actions'] = actions

    # def get_db(self,is_train):
    #
    #     mode = 1
    #     if not is_train:
    #         mode = 0
    #     sequences = self.db['sequences'][mode]
    #     frame_idx = self.db['frame_idx'][mode]
    #     seq_ids = self.db['seq_ids'][mode]
    #     actions = self.db['actions'][mode]
    #     db = {
    #         'data_path': self.data_path,
    #         'sequences': sequences,
    #         'frame_idx': frame_idx,
    #         'seq_ids': seq_ids,
    #         'actions': actions
    #     }
    #
    #     return db

    def _ntu_load_annotations(self):
        assert self.eval_mode in ['cs', 'cv'], \
            'Invalid evaluation mode {}'.format(self.eval_mode)
        dirty_data = ['S005C001P021R002A057',
                      'S010C002P016R001A053',
                      'S010C003P016R002A055',
                      'S010C003P017R001A055',
                      'S011C003P028R001A055',
                      'S011C003P038R001A055',
                      'S012C002P016R001A053',
                      'S012C003P016R002A055',
                      'S012C003P037R001A055',
                      'S016C003P040R002A053']
        cs_train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
                    25, 27, 28, 31, 34, 35, 38]
        cv_train = [2, 3]

        file_name = os.path.join(self.data_dir, 'ntu60_hrnet.pkl')

        seq_ids = [[], [], []]
        seqs = [[], [], []]
        frame_ids_list = [[], [], []]
        actions = [[], [], []]

        f = open(file_name, 'rb')
        info = pickle.load(f)
        f.close()
        annotations = info['annotations']
        with alive_bar(len(annotations)) as bar:
            for anno in annotations:
                bar()
                action = anno['label']
                sequence_id = anno['frame_dir']
                frame_dir = os.path.join(self.data_dir, 'images', sequence_id)
                s = int(sequence_id[1:4])
                if s > self.num_S:
                    continue
                if not os.path.isdir(frame_dir) or sequence_id in dirty_data:
                    continue
                if self.random_pick_half and random.random() < 0.5:
                    continue
                p = int(sequence_id[9:12])
                c = int(sequence_id[5:8])
                if self.eval_mode == 'cs':
                    # TRAN_MODE = 1 TEST_MODE = 0
                    mode = TRAIN_MODE if p in cs_train else TEST_MODE
                else:
                    mode = TRAIN_MODE if c in cv_train else TEST_MODE

                total_frames = os.listdir(frame_dir)
                total_frames.sort(key=lambda x: int(x[:-4]))
                frame_ids = [int(f[:-4]) for f in total_frames]
                # S001C001P003R001A059
                sequence = anno['keypoint']

                # A_joints, A_joints_vis = self._get_joints_vis(sequence[0])
                # A_bbox = np.zeros((len(A_joints), 4))
                # if len(sequence) == 2:
                #     B_joints, B_joints_vis = self._get_joints_vis(sequence[1])
                #     B_bbox = np.zeros((len(A_joints), 4))

                tongji = 0
                # for i in range(len(sequence[0])):  # every frame
                #     A_bbox[i] = get_gt_bbox(A_joints[i:i + 1], A_joints_vis[i:i + 1], self.read_image_size,
                #                             scale=self.bbox_scale, logkey=sequence_id)
                #     if len(sequence) == 2:
                #         if np.sum(B_joints_vis[i]) <= int(B_joints_vis.shape[1] * 0.8):
                #             B_joints[i] = A_joints[i]
                #             B_joints_vis[i] = A_joints_vis[i]
                #             B_bbox[i] = A_bbox[i]
                #
                #             tongji+=1
                #         else:
                #             B_bbox[i] = get_gt_bbox(B_joints[i:i + 1], B_joints_vis[i:i + 1], self.read_image_size,
                #                                     scale=self.bbox_scale, logkey=sequence_id)
                # if tongji>0:
                #     print(f'{sequence_id} B switch to A frames number is {tongji}')

                # joints *= self.ntu_scale_ratio  # (frames, joints, 2)

                """经测试 双人动作中 第一个人不可能出现某一帧全部骨骼点都invisible的情况，第二个人经常出现"""
                # bbox = get_gt_bbox(poses[i:i + 1, :, 0:2], visibles[i:i + 1], (w, h),
                #                    scale=cfg['DATASET']['CROP_SCALE'],
                #                    logkey=idx)
                # center, scale = bbox_to_objposwin(bbox)
                # if action>=49 and len(sequence)!=2:
                #     print(sequence_id,'action>=49 and len(sequence)!=2')
                # if action < 49 and len(sequence) != 1:
                #     print(sequence_id, 'action < 49 and len(sequence) != 1')

                seq_ids[mode].append(sequence_id)
                seqs[mode].append(sequence)
                frame_ids_list[mode].append(frame_ids)
                actions[mode].append(action)

        return seq_ids, seqs, frame_ids_list, actions
