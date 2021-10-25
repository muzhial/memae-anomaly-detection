from yacs.config import CfgNode as CN


__C = CN()

cfg = __C

__C.SUFFIX = ['.png', '.jpeg', '.jpg', '.tif', '.bmp']

__C.DATA = CN()
__C.DATA.DATA_ROOT = '/dataset/mz/outside_data/UCSD_Anomaly_Dataset.v1p2'
__C.DATA.DATA_NAME = 'UCSDped2'
__C.DATA.OUT_ROOT = 'processed/UCSD_P2_256'
__C.DATA.OUTSIZE = [256, 256]
__C.DATA.MAXS = 320

__C.TRAIN = CN()
__C.TRAIN.CLIP_LEN = 16
__C.TRAIN.OVERLAP_RATE = 0
__C.TRAIN.SKIP_STEP = 2  # 1
__C.TRAIN.CLIP_RNG = __C.TRAIN.CLIP_LEN * __C.TRAIN.SKIP_STEP
# __C.TRAIN.OVERLAP_SHIFT = __C.TRAIN.CLIP_LEN - 1
__C.TRAIN.OVERLAP_SHIFT = 10
