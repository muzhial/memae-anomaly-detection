import os

from tqdm import tqdm
import cv2
import numpy as np
import mmcv

from config import cfg


def gen_video_clip_idx(idxs):
    num_frames = len(idxs)
    start_idxs = np.array(range(
        0, num_frames, cfg.TRAIN.CLIP_RNG - cfg.TRAIN.OVERLAP_SHIFT))
    end_idxs = start_idxs + cfg.TRAIN.CLIP_RNG
    valid_idx = np.where(end_idxs < num_frames)
    results = []
    for i, start in enumerate(start_idxs[valid_idx]):
        results.append(list(range(start, end_idxs[i], cfg.TRAIN.SKIP_STEP)))
    results = np.array(results)
    assert results.shape[1] == cfg.TRAIN.CLIP_LEN, \
        f'clip len should be {cfg.TRAIN.CLIP_LEN}, but get {results.shape[1]}'
    return np.array(results)

def gen_label_index(split, data_path):
    out_path = os.path.join(data_path, str(split) + '_idx')
    mmcv.mkdir_or_exist(out_path)
    data_path = os.path.join(data_path, split)
    for frames_dir_name in tqdm(os.listdir(data_path)):
        frames_dir = os.path.join(data_path, frames_dir_name)
        frames_idx_path = os.path.join(out_path, frames_dir_name + '.npy')
        if os.path.isdir(frames_dir):
            frames_name = [n
                for n in os.listdir(frames_dir)
                    if os.path.splitext(n)[1] in cfg.SUFFIX]
            frames_name.sort()
            frames_idxs = gen_video_clip_idx(frames_name)
            np.save(frames_idx_path, frames_idxs)

def transform_image(img_file, resize=True):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    if resize:
        trans_img = cv2.resize(img, cfg.DATA.OUTSIZE)
    else:
        trans_img = img
    return trans_img

def prepare_data(split, data_path, out_path):
    data_path = os.path.join(data_path, split)
    out_path = os.path.join(out_path, split)
    mmcv.mkdir_or_exist(out_path)

    for f in tqdm(os.listdir(data_path)):
        frame_dir = os.path.join(data_path, f)
        if os.path.isdir(frame_dir):
            frame_dir_out = os.path.join(out_path, f)
            mmcv.mkdir_or_exist(frame_dir_out)

            for frame_name in os.listdir(frame_dir):
                if os.path.splitext(frame_name)[1] in cfg.SUFFIX:
                    frame_path = os.path.join(frame_dir, frame_name)
                    frame_path_out = os.path.join(frame_dir_out, frame_name)
                    trans_frame = transform_image(frame_path)
                    cv2.imwrite(frame_path_out, trans_frame)

def main():
    data_path = os.path.join(cfg.DATA.DATA_ROOT, cfg.DATA.DATA_NAME)
    out_path = os.path.join(cfg.DATA.DATA_ROOT, cfg.DATA.OUT_ROOT)

    # prepare_data('Train', data_path, out_path)

    gen_label_index('Train', out_path)


if __name__ == '__main__':
    main()
