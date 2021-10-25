import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import data
from options.training_options import TrainOptions
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem
from models import EntropyLossEncap
from tools import cfg


###
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

###
utils.seed(opt.Seed)
if(opt.IsDeter):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

######
model_setting = utils.get_model_setting(opt)
print('Setting: %s' % (model_setting))

model_setting = 'out1'

############
batch_size_in = opt.BatchSize
learning_rate = opt.LR
max_epoch_num = opt.EpochNum

chnum_in_ = opt.ImgChnNum      # channel number of the input images
framenum_in_ = opt.FrameNum  # num of frames in a video clip
mem_dim_in = opt.MemDim
entropy_loss_weight = opt.EntropyLossWeight
sparse_shrink_thres = opt.ShrinkThres

img_crop_size = 0

print('bs=%d, lr=%f, entrloss=%f, shr=%f, memdim=%d' %
    (batch_size_in,
     learning_rate,
     entropy_loss_weight,
     sparse_shrink_thres,
     mem_dim_in))
############
## data path
data_root = os.path.join(opt.DataRoot, opt.Dataset)
tr_data_frame_dir = os.path.join(data_root, 'rawframes')
tr_data_idx_dir = os.path.join(data_root, 'rawframes_idx')

# ############ model saving dir path
saving_root = opt.ModelRoot
saving_model_path = os.path.join(saving_root, 'model_' + model_setting + '/')
utils.mkdir(saving_model_path)

# ### tblog
# if(opt.IsTbLog):
#     log_path = os.path.join(saving_root, 'log_'+model_setting + '/')
#     utils.mkdir(log_path)
#     tb_logger = utils.Logger(log_path)

# ##
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

# ###### data
video_dataset = data.VideoDataset(
    tr_data_idx_dir,
    tr_data_frame_dir,
    transform=frame_trans,
    cfg=cfg)
tr_data_loader = DataLoader(video_dataset,
                            batch_size=batch_size_in,
                            shuffle=True,
                            num_workers=opt.NumWorker)

# ###### model
if(opt.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(chnum_in_,
                                mem_dim_in,
                                shrink_thres=sparse_shrink_thres)
else:
    model = []
    print('Wrong model name.')
model.apply(utils.weights_init)

# #########
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)
tr_recon_loss_func = nn.MSELoss().to(device)
tr_entropy_loss_func = EntropyLossEncap().to(device)
tr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ##
data_loader_len = len(tr_data_loader)
textlog_interval = opt.TextLogInterval
snap_save_interval = opt.SnapInterval
save_check_interval = opt.SaveCheckInterval
tb_img_log_interval = opt.TBImgLogInterval
global_ite_idx = 0 # for logging
for epoch_idx in range(0, max_epoch_num):
    for batch_idx, frames in enumerate(tr_data_loader):
        frames = frames.to(device)
        if (opt.ModelName == 'MemAE'):
            recon_res = model(frames)
            recon_frames = recon_res['output']
            att_w = recon_res['att']
            loss = tr_recon_loss_func(recon_frames, frames)
            recon_loss_val = loss.item()
            entropy_loss = tr_entropy_loss_func(att_w)
            entropy_loss_val = entropy_loss.item()
            loss = loss + entropy_loss_weight * entropy_loss
            loss_val = loss.item()

            tr_optimizer.zero_grad()
            loss.backward()
            tr_optimizer.step()

        if((batch_idx % textlog_interval) == 0):
            print('[epoch %d/%d, bt %d/%d] loss=%f, rc_losss=%f, ent_loss=%f'
                % (
                    epoch_idx,
                    max_epoch_num,
                    batch_idx,
                    data_loader_len,
                    loss_val, recon_loss_val, entropy_loss_val) )
        # if((global_ite_idx % snap_save_interval)==0):
        #     torch.save(model.state_dict(), '%s/%s_snap.pt' % (saving_model_path, model_setting) )
        # global_ite_idx += 1
    if (epoch_idx % save_check_interval) == 0 and epoch_idx != 0:
        torch.save(model.state_dict(),
                   '%s/%s_epoch_%04d.pt' % (
                       saving_model_path, model_setting, epoch_idx) )

torch.save(model.state_dict(),
           '%s/%s_epoch_%04d_final.pt' % (
               saving_model_path, model_setting, epoch_idx) )
