import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import data_utils
import pathlib


class H36M(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, actions=None, split=0, miss_rate=0.2, all_data=False,
                 joints=32):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 validation, 2 testing
        :param sample_rate:
        """
        # !Luca: changed path to data
        self.path_to_data = os.path.join(data_dir, 'h36m')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.miss_rate = miss_rate
        self.sample_rate = 2
        self.p3d = {}
        self.params = {}
        self.masks = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions

        print('Joints: ', joints, 'Path: ', pathlib.Path(__file__).parent.resolve())

        if joints == 17:
            self.dim_used = np.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42,
                 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83])
        elif joints == 22:
            self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                                 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                                 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        else:
            self.dim_used = np.arange(96)

        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                if not all_data:
                    if key >= 30:
                        continue
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        the_sequence[:, 0:6] = 0
                        p3d = data_utils.expmap2xyz_torch(the_sequence)

                        # self.p3d[(subj, action, subact)] = p3d.view(num_frames, -1).cpu().data.numpy()
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                        # self.p3d[key] = self.p3d[key][:, dim_used]

                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)

                        # tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                        print(num_frames, len(valid_frames))
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    # self.p3d[key] = self.p3d[key][:, dim_used]

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    # self.p3d[key + 1] = self.p3d[key + 1][:, dim_used]

                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                                   input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        pose = self.p3d[key][fs]
        observed = pose.copy() / 1000.0

        mask = np.zeros((pose.shape[0], pose.shape[1])) #! Create the mask
        mask[0:self.in_n, :] = 1 #! history part should be visible
        mask[self.in_n:self.in_n + self.out_n, :] = 0 #! future has to be masked and forcasted

        data = {
            "pose": observed[:, self.dim_used], #! Pose to use
            "pose_32": pose, #! Pose with irrelevant joints (complete)
            "mask": mask.copy()[:, self.dim_used], #! Temporal Mask to avoid leaking the future
            "timepoints": np.arange(self.in_n + self.out_n) #! ? [Probably the ranking of timeframes in a pose]
        }

        return data
