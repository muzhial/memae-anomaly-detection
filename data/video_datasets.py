import os
import glob

import numpy as np
import cv2
# import scipy.io as sio
# from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    N x C x T x H x W
    """

    def __init__(self,
                 idx_root,
                 frame_root,
                 transform=None,
                 idx_suffix='.npy',
                 cfg=None):
        self.cfg = cfg
        self.idx_root = idx_root
        self.frame_root = frame_root
        self.transform = transform

        self.idx_file_list = self.load_idx_file(
            self.idx_root, idx_suffix, self.frame_root)

        self.idx_file_list = self.idx_file_list[:2000]

    def load_idx_file(self, idx_root, idx_suffix, frame_root):
        idx_file_list = []
        for idx_label_file in glob.glob(idx_root + f'/*{idx_suffix}'):
            idx_file_name = os.path.basename(idx_label_file)
            idx_array = np.load(idx_label_file)
            frame_dir = os.path.join(
                frame_root, os.path.splitext(idx_file_name)[0])
            frame_file_list = [n
                for n in os.listdir(frame_dir)
                    if os.path.splitext(n)[1] in self.cfg.SUFFIX]
            frame_file_list.sort()
            for row in idx_array:
                clip_file_list = []
                for col in row:
                    clip_file_list.append(
                        os.path.join(frame_dir, frame_file_list[col]))
                idx_file_list.append(clip_file_list)

        return idx_file_list

    def __len__(self):
        return len(self.idx_file_list)

    def __getitem__(self, index):
        clip_files = self.idx_file_list[index]
        frames = torch.stack(
            [self.transform(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
                for frame in clip_files],
            dim=1)
        return frames

# Video index files are organized in correlated sub folders.
# N x C x T x H x W
# class VideoDataset(Dataset):
#     def __init__(self, idx_root, frame_root, use_cuda=False, transform=None):
#         # dir name
#         self.idx_root = idx_root
#         self.frame_root = frame_root

#         # video_name_list, subdir names
#         self.video_list = [name for name in os.listdir(self.idx_root) \
#                               if os.path.isdir(os.path.join(self.idx_root, name))]
#         self.video_list.sort()

#         #
#         self.idx_path_list = []
#         for ite_vid in range(len(self.video_list)):
#             video_name = self.video_list[ite_vid]
#             # idx file name list
#             idx_file_name_list = [name for name in os.listdir(os.path.join(self.idx_root, video_name)) \
#                               if os.path.isfile(os.path.join(self.idx_root, video_name, name))]
#             idx_file_name_list.sort()
#             # idx file path list
#             idx_file_list = [self.idx_root + '/' + video_name + '/' + file_name for file_name in idx_file_name_list]
#             # merger lists
#             self.idx_path_list = self.idx_path_list + idx_file_list
#         self.idx_num = len(self.idx_path_list)
#         self.use_cuda = use_cuda
#         self.transform = transform

#     def __len__(self):
#         return self.idx_num

#     def __getitem__(self, item):
#         """ get a video clip with stacked frames indexed by the (idx) """
#         idx_path = self.idx_path_list[item] # idx file path
#         idx_data = sio.loadmat(idx_path)    # idx data
#         v_name = idx_data['v_name'][0]  # video name
#         frame_idx = idx_data['idx'][0, :]  # frame index list for a video clip

#         v_dir = self.frame_root + v_name

#         tmp_frame = io.imread(os.path.join(v_dir, ('%03d' % frame_idx[0]) + '.jpg'))
#         tmp_frame_shape = tmp_frame.shape
#         frame_cha_num = len(tmp_frame_shape)
#         # h = tmp_frame_shape[0]
#         # w = tmp_frame_shape[1]
#         if frame_cha_num==3:
#             c = tmp_frame_shape[2]
#         elif frame_cha_num==2:
#             c = 1
#         # each sample is concatenation of the indexed frames
#         if self.transform:
#             if c==3:
#                 frames = torch.cat([self.transform(
#                     io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'))).unsqueeze(1) for i
#                                     in frame_idx], 1)
#             elif c==1:
#                 frames = torch.cat([self.transform(
#                     np.expand_dims(io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')), axis=2)).unsqueeze(1) for i
#                                     in frame_idx], 1)
#         else:
#             tmp_frame_trans = transforms.ToTensor() # trans Tensor
#             if c==3:
#                 frames = torch.cat([tmp_frame_trans(
#                     io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'))).unsqueeze(1) for i
#                                     in frame_idx], 1)
#             elif c==1:
#                 frames = torch.cat([tmp_frame_trans(
#                     np.expand_dims(io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'), axis=2))).unsqueeze(1) for i
#                                     in frame_idx], 1)
#         return item, frames


# All video index files are in one dir.
# N x C x T x H x W

class VideoDatasetTest(Dataset):
    """
    N x C x T x H x W
    """

    def __init__(self,
                 idx_root,
                 frame_root,
                 transform=None,
                 idx_suffix='.npy',
                 cfg=None):
        self.cfg = cfg
        self.idx_root = idx_root
        self.frame_root = frame_root
        self.transform = transform

        self.idx_file_list = self.load_idx_file(
            self.idx_root, idx_suffix, self.frame_root)

        frame_start = 2500
        frame_end = frame_start + 16
        self.idx_file_list = self.idx_file_list[frame_start:frame_end]

    def load_idx_file(self, idx_root, idx_suffix, frame_root):
        idx_file_list = []
        for idx_label_file in glob.glob(idx_root + f'/*{idx_suffix}'):
            idx_file_name = os.path.basename(idx_label_file)
            idx_array = np.load(idx_label_file)
            frame_dir = os.path.join(
                frame_root, os.path.splitext(idx_file_name)[0])
            frame_file_list = [n
                for n in os.listdir(frame_dir)
                    if os.path.splitext(n)[1] in self.cfg.SUFFIX]
            frame_file_list.sort()
            for row in idx_array:
                clip_file_list = []
                for col in row:
                    clip_file_list.append(
                        os.path.join(frame_dir, frame_file_list[col]))
                idx_file_list.append(clip_file_list)

        return idx_file_list

    def __len__(self):
        return len(self.idx_file_list)

    def __getitem__(self, index):
        clip_files = self.idx_file_list[index]
        add_noise = True
        noise_frame_list = [3, 6, 8, 14]

        # frames = torch.stack(
        #     [self.transform(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
        #         for frame in clip_files],
        #     dim=1)

        frames = []
        for frame in clip_files:
            img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
            if add_noise and index in noise_frame_list:
                img[206:216, 206:216] = 0
            frames.append(self.transform(img))
        frames = torch.stack(frames, dim=1)

        return frames
