import os
import os.path as osp
import argparse
import glob
import sys
import warnings
from multiprocessing import Lock, Pool

from tqdm import tqdm
import numpy as np
import mmcv

from config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        default='/dataset/mz/outside_data/fault_vid/out1.mp4')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='/dataset/mz/outside_data/fault_vid/rawframes')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=8,
        help='number os workers to build rawframes')
    parser.add_argument(
        '--out_format',
        type=str,
        default='jpg',
        choices=['jpg', 'h5', 'png'])
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm'])
    parser.add_argument(
        '--new_height',
        type=int,
        default=512)
    parser.add_argument(
        '--new_width',
        type=int,
        default=512)
    args = parser.parse_args()

    return args

def init(lock_):
    global lock
    lock = lock_

def extract_frame(*vid_item):
    full_path, out_full_path = vid_item
    run_success = -1

    try:
        vr = mmcv.VideoReader(full_path)
        pbar = tqdm(total=int(len(vr)))
        for i, vr_frame in tqdm(enumerate(vr)):
            if vr_frame is not None:
                h, w, _ = np.shape(vr_frame)
                out_img = mmcv.imresize(
                    vr_frame, (args.new_width, args.new_height))
                mmcv.imwrite(out_img,
                             f'{out_full_path}/img_{i + 1:06d}.jpg')
            else:
                warnings.warn(
                    'Length inconsistent.'
                    f'Early stop with {i + 1} out of {len(vr)} frames.'
                )
                break
            pbar.update(1)
        pbar.close()

        run_success = 0
    except Exception:
        run_success = -1

    if run_success == 0:
        print(f'done')
        sys.stdout.flush()

        # lock.acquire()
        # can write something
        # lock.release()
    else:
        print(f'got something wrong')
        sys.stdout.flush()

    return True


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


if __name__ == '__main__':
    args = parse_args()

    args.out_dir = osp.join(
        args.out_dir,
        osp.splitext(osp.basename(args.src_dir))[0])
    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    # lock = Lock()
    # pool = Pool(args.num_works, initializer=init, initargs=(lock, ))
    # pool.map(
    #     extract_frame,
    #     zip()
    # )
    # pool.close()
    # pool.join()

    # extract_frame(args.src_dir, args.out_dir)

    gen_label_index('rawframes', '/dataset/mz/outside_data/fault_vid')
