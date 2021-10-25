import os
import utils
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import data
from options.testing_options import TestOptions
import utils
import time
import cv2
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from tools import cfg


###
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

###
batch_size_in = opt.BatchSize  # 1
chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum    # frame number of the input images in a video clip
mem_dim_in = opt.MemDim
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

######
model_setting = utils.get_model_setting(opt)

model_setting = 'out1'

## data path
data_root = os.path.join(opt.DataRoot, opt.Dataset)
data_frame_dir = os.path.join(data_root, 'rawframes')
data_idx_dir = os.path.join(data_root, 'rawframes_idx')

############ model path
model_root = opt.ModelRoot
if(opt.ModelFilePath):
    model_path = opt.ModelFilePath
else:
    model_path = os.path.join(model_root, model_setting + '.pt')

### test result path
te_res_root = opt.OutRoot
te_res_path = te_res_root + '/' + 'res_' + model_setting
utils.mkdir(te_res_path)

###### loading trained model
if (opt.ModelName == 'AE'):
    model = AutoEncoderCov3D(chnum_in_)
elif(opt.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(chnum_in_, mem_dim_in, shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong Name.')

##
model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

##
if(chnum_in_==1):
    norm_mean = [0.5]
    norm_std = [0.5]
elif(chnum_in_==3):
    norm_mean = (0.5, 0.5, 0.5)
    norm_std = (0.5, 0.5, 0.5)

frame_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

show_image = False

with torch.no_grad():
    video_dataset = data.VideoDatasetTest(
        data_idx_dir,
        data_frame_dir,
        frame_trans,
        cfg=cfg)

    video_dataloader = DataLoader(video_dataset,
                                  batch_size=batch_size_in,
                                  shuffle=False)
    recon_error_list = []

    for batch_idx, frames in enumerate(video_dataloader):
        frames = frames.to(device)

        if (opt.ModelName == 'AE'):
            recon_frames = model(frames)
            ###### calculate reconstruction error (MSE)
            recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
            input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
            r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
            # recon_error = np.mean(sum(r**2)**0.5)
            recon_error = np.mean(r ** 2)  # **0.5
            recon_error_list += [recon_error]
        elif (opt.ModelName == 'MemAE'):
            recon_res = model(frames)
            recon_frames = recon_res['output']
            if show_image:
                unorm_recon_frames = unorm_trans(recon_frames)
                # ndarray(T, C, H, W)
                recon_np = utils.vframes2imgs(recon_frames, step=1, batch_idx=0)
                assert len(recon_np.shape) == 4, 'recon_np should be 4-d'
                recon_np = np.transpose(recon_np, (0, 2, 3, 1))
                for i in range(recon_np.shape[0]):
                    recon_frame = (recon_np[i] * 255).astype(np.uint8)
                    recon_frame = cv2.cvtColor(recon_frame, cv2.COLOR_GRAY2BGR)
                    # print(recon_frame.dtype, recon_frame.min(), recon_frame[i].max())
                    cv2.imwrite(f'./results/res_out1/{i}.png', recon_frame)

            r = recon_frames - frames
            r = utils.crop_image(r, img_crop_size)
            sp_error_map = torch.sum(r**2, dim=1) ** 0.5
            s = sp_error_map.size()
            sp_error_vec = sp_error_map.view(s[0], -1)
            recon_error = torch.mean(sp_error_vec, dim=-1)
            recon_error_list += recon_error.cpu().tolist()
        else:
            recon_error = -1
            print('Wrong ModelName.')
    recon_error_list_round = [round(f, 4) for f in recon_error_list]
    print(recon_error_list_round)
    np.save(os.path.join(te_res_path, 'out1' + '.npy'), recon_error_list)

## evaluation
# utils.eval_video(data_root, te_res_path, is_show=False)
