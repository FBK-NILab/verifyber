#!/usr/bin/env python

from __future__ import print_function

import argparse
import configparser
import glob
import json
import os
import subprocess
import warnings
from os import path as osp
from os.path import basename as osbn
from time import time
import random

import ants
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from dipy.tracking.streamline import transform_streamlines
from torch_geometric.data import Batch as gBatch
try:
    from torch_geometric.loader import DataListLoader as gDataLoader
except:
    from torch_geometric.data import DataListLoader as gDataLoader
from tqdm import tqdm

from datasets import TractDataset
# from utils.data import selective_loader as sload
from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast
from utils.data.data_utils import (resample_streamlines, slr_with_qbx_partial,
                                   tck2trk, trk2tck)
from utils.data.transforms import TestSampling
from utils.general_utils import get_cfg_value
from utils.model_utils import get_model

# os.environ["DEVICE"] = torch.device(
#     'cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 10

# for repro
random.seed(SEED)
np.random.seed(SEED)
rs = np.random.RandomState(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except:
    torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility



script_dir = osp.dirname(osp.realpath(__file__))
tmp_dir = 'tmp_tractogram_filtering'

mni_fn_dict = {
    'fa': f'{script_dir}/data/standard/FSL_HCP1065_FA_1mm.nii.gz',
    't1': f'{script_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz',
    'slr': f'{script_dir}/data/standard/ZHANG_atlas_mni_centroids-qbx.trk'
}

tx_type_dict = {
    'lin': 'TRSAA',
    'fast': 'SyNRA',  # 'antsRegistrationSyNQuick[s]' with ants v0.2.6
    'slow': 'SyNCC',
}


def get_gpu_free_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are free memory as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ],
                                     encoding='utf-8')
    # Convert lines into a dictionary
    gpu_used_memory = [int(x) for x in result.strip().split('\n')]
    n_gpus = len(gpu_used_memory)
    gpu_free_memory = []
    for i in range(n_gpus):
        tot_mem = torch.cuda.get_device_properties(i).total_memory
        tot_mem = int(tot_mem / 1024**2)
        gpu_free_memory.append(tot_mem - gpu_used_memory[i])

    gpu_free_memory_map = dict(zip(range(n_gpus), gpu_free_memory))
    return gpu_free_memory_map


def get_max_batchsize(curr_device):
    free_mem = round(get_gpu_free_memory_map()[curr_device] / 1024, 2)  # in GB
    print(f'{free_mem} GB available on current GPU')
    return  int(free_mem / 4 * 10000)


def tract2standard_sl_based(t_fn, fixed_fn, t_std_fn):

    print('Loading tractogram...')
    centroids_fix = load_streamlines_fast(fixed_fn, container='ArraySequence')
    t_mov = load_streamlines_fast(t_fn, container='ArraySequence')

    print(f'Registration using SLR-QBX...')
    t_mov_aligned, affine_tx, centroids_mov = slr_with_qbx_partial(
        centroids_fix,
        t_mov,
        x0='affine',
        rm_small_clusters=300,
        verbose=True,
        greater_than=50,
        less_than=200,
        qbx_thr=[40, 30, 20, 15],
        nb_pts=20,
        rng=rs)
    
    print(f'Saving aligned tractogram...')
    t_mov_aligned = nib.streamlines.Tractogram(t_mov_aligned,
                                               affine_to_rasmm=np.eye(4))
    nib.streamlines.save(t_mov_aligned, t_std_fn)

    centroids_mov_aligned = transform_streamlines(centroids_mov, affine_tx)
    centroids_mov_aligned = nib.streamlines.Tractogram(
        centroids_mov_aligned, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(centroids_mov_aligned,
                         f'{tmp_dir}/centroids_mov_mni.trk')

    np.save(f'{tmp_dir}/sub2mni_affine_for_streamlines.npy', affine_tx)

    return t_std_fn


def tract2standard_img_based(t_fn,
                             t1_fn,
                             fixed_fn,
                             t_std_fn='tract_standard.tck',
                             trans_type='fast'):

    tx_type = tx_type_dict[trans_type]

    print(f'registration using ANTs {tx_type}...')
    fixed = ants.image_read(fixed_fn)
    moving = ants.image_read(t1_fn)

    to_invert = [True, False]

    # this is a workaround to emulate antsRegistrationSyNQuick.sh.
    # Unfortunately it is not possible to equally emulate the script.
    # There are differences in terms of parameters (shrink factor and num of
    # iterations) in the rigid and in the affine registration
    if tx_type == 'SyNRA':
        # values taken from https://github.com/ANTsX/ANTs/blob/952e7918b47385ebfb730f9c844977762b8437f8/Scripts/antsRegistrationSyNQuick.sh#L455
        # Notes:
        # 1. syn_metric and num_of_bins (syn_sampling) are the same as default:
        # "mattes" and 32 respectively
        # 2. the three values that configure the SyN[x,x,x] optimization are
        # respectively grad_step, flow_sigma, and total_sigma
        # 3. syn_iterations correspond to reg_iterations
        # 4. smoothing sigmas and shrink factor are automatically set inside the
        # function. As desired they are set to be: "3x2x1x0vox" and "8x4x2x1"
        # respectively
        mytx = ants.registration(fixed=fixed,
                                 moving=moving,
                                 type_of_transform=tx_type,
                                 reg_iterations=(100, 70, 50, 0),
                                 grad_step=0.1,
                                 flow_sigma=3,
                                 total_sigma=0,
                                 outprefix=f'{tmp_dir}/ants_tract2standard',
                                 random_seed=SEED)
    elif tx_type == 'TRSAA':
        to_invert = [True]
        mytx = ants.registration(fixed=fixed,
                                 moving=moving,
                                 type_of_transform=tx_type,
                                 reg_iterations=(1200, 1200, 1200, 0),
                                 outprefix=f'{tmp_dir}/ants_tract2standard',
                                 random_seed=SEED)
    else:
        mytx = ants.registration(fixed=fixed,
                                 moving=moving,
                                 type_of_transform=tx_type,
                                 outprefix=f'{tmp_dir}/ants_tract2standard',
                                 random_seed=SEED)

    # store warped struct for registration visual check
    ants.image_write(mytx['warpedmovout'], f'{tmp_dir}/struct_warped.nii.gz')

    print('correcting warp to mrtrix convention...')
    os.system(f'warpinit {fixed_fn} {tmp_dir}/ID_warp[].nii.gz -force')

    for i in range(3):
        temp_warp = ants.image_read(f'{tmp_dir}/ID_warp{i}.nii.gz')
        temp_warp = ants.apply_transforms(fixed=moving,
                                          moving=temp_warp,
                                          transformlist=mytx['invtransforms'],
                                          whichtoinvert=to_invert,
                                          defaultvalue=2147483647)
        ants.image_write(temp_warp, f'{tmp_dir}/mrtrix_warp{i}.nii.gz')

    os.system(f'warpcorrect {tmp_dir}/mrtrix_warp[].nii.gz ' +
              f'{tmp_dir}/mrtrix_warp_cor.nii.gz ' +
              '-marker 2147483647 -tolerance 0.0001 -force')

    print('applying warp to tractogram...')
    os.system(
        f'tcktransform {t_fn} {tmp_dir}/mrtrix_warp_cor.nii.gz {t_std_fn} ' +
        '-force -nthreads 0')

    return t_std_fn


def get_sample(data):
    gdata = gBatch().from_data_list([data['points']])
    gdata = gdata.to(DEVICE)
    gdata.batch = gdata.bvec.clone()
    del gdata.bvec
    gdata['lengths'] = gdata['lengths'][0].item()

    return gdata


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        nargs='?',
                        default=f'{script_dir}/run_config.json',
                        help='The tag for the configuration file.')
    args = parser.parse_args()

    ## load config
    t0_global = time()
    print('reading arguments')
    cfg = json.load(open(args.config))
    print(cfg)

    cfg['n_classes'] = 2
    cfg['trk'] = osp.abspath(cfg['trk'])
    move_tract = cfg['warp'] != ''
    img_type = 'fa' if cfg['fa'] != '' else 't1'

    tck_fn = f'{tmp_dir}/input/tract.tck'
    tck_mni_fn = f'{tck_fn[:-4]}_mni.tck'
    trk_mni_fn = cfg['trk']
    trk_fn = f'{tmp_dir}/input/tract_mni_resampled.trk'

    in_dir = f'{tmp_dir}/input'
    os.makedirs(in_dir, exist_ok=True)

    skip = False
    if osp.exists(trk_fn):
        warnings.warn('Found a precomputed tractogram, using it')
        skip = True

    ## compute warp to mni and move tract if needed
    if move_tract and not skip:

        if cfg['warp'] == 'slr':
            t0 = time()
            trk_mni_fn = tract2standard_sl_based(cfg['trk'], mni_fn_dict['slr'],
                                                 f'{tck_mni_fn[:-4]}.trk')
            print(f'done in {time()-t0} sec')
        else:
            if not osp.exists(tck_fn):
                t0 = time()
                print('convert trk to tck...')
                trk2tck(cfg['trk'], out_fn=tck_fn)
                print(f'done in {time()-t0} sec')

            t0 = time()
            tract2standard_img_based(tck_fn,
                                     osp.abspath(cfg[img_type]),
                                     mni_fn_dict[img_type],
                                     t_std_fn=tck_mni_fn,
                                     trans_type=cfg['warp'])
            print(f'done in {time()-t0} sec')

    ## resample trk to 16points if needed
    if cfg['resample_points'] and not skip:
        t0 = time()
        print('loading tractogram for resampling...')
        if not osp.exists(tck_mni_fn):
            streamlines = load_streamlines_fast(trk_mni_fn)
        else:
            streamlines = nib.streamlines.load(tck_mni_fn).streamlines
        print(f'done in {time()-t0} sec')

        t0 = time()
        print('streamlines resampling...')
        streamlines = resample_streamlines(streamlines)
        print(f'done in {time()-t0} sec')
        t0 = time()
        print('saving resampled tractogram...')
        resampled_t = nib.streamlines.Tractogram(streamlines,
                                                 affine_to_rasmm=np.eye(4))
        nib.streamlines.save(resampled_t, trk_fn)
        print(f'done in {time()-t0} sec')

    if not osp.exists(trk_fn):
        if not osp.exists(tck_mni_fn):
            print('The tractogram loaded is already compatible with the model')
            os.system(f'''ln -sf {cfg['trk']} {trk_fn}''')
        else:
            t0 = time()
            print('convert warped tck to trk...')
            tck2trk(tck_mni_fn, mni_fn, out_fn=trk_fn)
            print(f'done in {time()-t0} sec')

    ## run inference
    print(f'launching inference using {DEVICE}...')
    exp = f'{script_dir}/checkpoints/{cfg["model"]}'

    cfg_parser = configparser.ConfigParser()
    cfg_parser.read(exp + '/config.txt')

    for section in cfg_parser.keys():
        for name, value in cfg_parser.items(section):
            cfg[name] = get_cfg_value(value)

    cfg['with_gt'] = False
    cfg['weights_path'] = ''
    cfg['exp_path'] = exp

    # check available memory to decide how many streams sample
    curr_device = torch.cuda.current_device()
    cfg['fixed_size'] = get_max_batchsize(curr_device)
    print(f'set batch size to {cfg["fixed_size"]}')

    dataset = TractDataset(trk_fn,
                           transform=TestSampling(cfg['fixed_size']),
                           return_edges=True,
                           split_obj=True)

    dataloader = gDataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    classifier = get_model(cfg)

    if DEVICE == 'cuda':
        torch.cuda.set_device(DEVICE)
        torch.cuda.current_device()

    if cfg['weights_path'] == '':
        cfg['weights_path'] = glob.glob(cfg['exp_path'] + '/best*')[0]
    state = torch.load(cfg['weights_path'], map_location=DEVICE)

    classifier.load_state_dict(state)
    classifier.to(DEVICE)
    classifier.eval()

    preds = []
    probas = []
    with torch.no_grad():
        j = 0
        i = 0
        while j < len(dataset):
            t0 = time()
            print(f'processing subject {j}...')
            consumed = False
            data = dataset[j]
            obj_pred = np.zeros(data['obj_full_size'])
            obj_proba = np.zeros(data['obj_full_size'])
            prog_bar = tqdm(total=len(dataset.remaining[j]))
            while not consumed:

                points = get_sample(data)
                batch = points.batch

                logits = classifier(points)

                pred = F.log_softmax(logits, dim=-1)
                pred_choice = pred.data.max(1)[1].int()

                obj_pred[data['obj_idxs']] = pred_choice.cpu().numpy()
                obj_proba[data['obj_idxs']] = F.softmax(
                    logits, dim=-1)[:, 0].cpu().numpy()

                prog_bar.update(len(data['obj_idxs']))
                if len(dataset.remaining[j]) == 0:
                    consumed = True
                    break
                data = dataset[j]
                i += 1

            preds.append(obj_pred)
            probas.append(obj_proba)

            j += 1
            print(f'done in {time()-t0} sec')
            prog_bar.close()

        ## save predictions
        out_dir = f'{tmp_dir}/output'
        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        print(f'saving predictions...')
        for pred in preds:
            idxs_P = np.where(pred == 1)[0]
            np.savetxt(f'{out_dir}/idxs_plausible.txt', idxs_P, fmt='%d')
            idxs_nonP = np.where(pred == 0)[0]
            np.savetxt(f'{out_dir}/idxs_non-plausible.txt',
                       idxs_nonP,
                       fmt='%d')
            if cfg['return_trk']:
                hdr = nib.streamlines.load(cfg['trk'], lazy_load=True).header
                streams, lengths = load_streamlines_fast(cfg['trk'], container='array_flat')
                streamlines = np.split(streams, np.cumsum(lengths[:-1]))
                streamlines = np.array(streamlines, dtype=np.object)[idxs_P]
                out_t = nib.streamlines.Tractogram(streamlines,
                                                   affine_to_rasmm=np.eye(4))
                out_t_name = osbn(cfg['trk'])[:-4] + '_filtered.trk'
                out_t_fn = f'''{out_dir}/{out_t_name}'''
                nib.streamlines.save(out_t, out_t_fn, header=hdr)
                print(f'saved {out_t_fn}')
        print(f'End')
        print(f'Duration: {(time()-t0_global)/60} min')
