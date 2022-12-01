from __future__ import print_function
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm

import nibabel as nib
import glob
import time
from utils.data.selective_loader import load_selected_streamlines, load_selected_streamlines_uniform_size
from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
import functools


class HCP20Dataset(gDataset):
    def __init__(self,
                 sub_file,
                 root_dir,
                 #k=4,
                 same_size=False,
                 act=True,
                 fold_size=None,
                 transform=None,
                 distance=None,
                 self_loops=None,
                 with_gt=True,
                 return_edges=False,
                 split_obj=False,
                 train=True,
                 load_one_full_subj=False,
                 standardize=False,
                 centering=False,
                 labels_dir=None,
                 permute=False):
        """
        Args:
            root_dir (string): root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        with open(sub_file) as f:
            subjects = [s.strip() for s in f.readlines()]
        self.subjects = subjects
        #self.k = k
        self.transform = transform
        self.distance = distance
        self.fold_size = fold_size
        self.act = act
        self.self_loops = self_loops
        self.with_gt = with_gt
        self.return_edges = return_edges
        self.fold = []
        self.n_fold = 0
        self.train = train
        self.load_one_full_subj = load_one_full_subj
        self.same_size = same_size
        self.standardize = standardize
        self.centering = centering
        self.permute = permute
        if fold_size is not None:
            self.load_fold()
        if train:
            split_obj=False
        if split_obj:
            self.remaining = [[] for _ in range(len(subjects))]
        self.split_obj = split_obj
        if with_gt:
            self.labels = []
            for sub in subjects:
                label_sub_dir = os.path.join(self.root_dir.rsplit('/',1)[0], labels_dir ,'sub-%s' % sub)
                label_file = os.path.join(label_sub_dir, 'sub-%s_var-HCP_labels_gt20mm.npy' % (sub))
                #label_file = os.path.join(label_sub_dir, 'sub-%s_CSD5TT8_weight.npy' % (sub))
                if label_file[-4:] == '.txt':
                    self.labels.append(np.loadtxt(label_file))
                else:
                    self.labels.append(np.load(label_file))


    def __len__(self):
        if self.load_one_full_subj:
            return len(self.full_subj[0])
        return len(self.subjects)

    def __getitem__(self, idx):
        fs = self.fold_size
        if fs is None:
            #print(self.subjects[idx])
            #t0 = time.time()
            item = self.getitem(idx)
            #print('get item time: {}'.format(time.time()-t0))
            return item

        fs_0 = (self.n_fold * fs)
        idx = fs_0 + (idx % fs)

        return self.data_fold[idx]


    def load_fold(self):
        print('loading fold')
        fs = self.fold_size
        fs_0 = self.n_fold * fs
        #t0 = time.time()
        print('Loading fold')
        self.data_fold = [self.getitem(i) for i in range(fs_0, fs_0 + fs)]
        #print('time needed: %f' % (time.time()-t0))

    def getitem(self, idx):
        sub = self.subjects[idx]
        #t0 = time.time()
        sub_dir = os.path.join(self.root_dir, 'sub-%s' % sub)
        T_file = os.path.join(sub_dir, 'sub-%s_var-HCP_full_tract_gt20mm.trk' % (sub))
        T = nib.streamlines.load(T_file, lazy_load=True)
        #print('\tload lazy T %f' % (time.time()-t0))
        #t0 = time.time()
        gt = self.labels[idx]
        #print('\tload gt %f' % (time.time()-t0))
        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))
            sample = {'points': np.array(list(self.remaining[idx]))}
            if self.with_gt:
                sample['gt'] = gt[list(self.remaining[idx])]
        else:
            #sample = {'points': np.arange(T.header['nb_streamlines'])}
            #if self.with_gt:
            #sample['gt'] = gt
            sample = {'points': np.arange(T.header['nb_streamlines']), 'gt': gt}
        #print(sample['name'])

        #t0 = time.time()
        if self.transform:
            sample = self.transform(sample)
        #print('\ttime sampling %f' % (time.time()-t0))

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']
            #sample['streamlines'] = T.streamlines

        sample['name'] = T_file.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = sub_dir

        n = len(sample['points'])
        #t0 = time.time()
        if self.same_size:
            # streams, lengths = load_selected_streamlines_uniform_size(T_file,
                                                    # sample['points'].tolist())
            streams, lengths = load_streamlines_fast(T_file,
                                                    sample['points'].tolist())
            if self.centering:
                streams_centered = streams.reshape(-1, lengths[0], 3)
                streams_centered -= streams_centered.mean(axis=1)[:,None,:]
                streams = streams_centered.reshape(-1,3)
            if self.permute:
                # import ipdb; ipdb.set_trace()
                streams_perm = self.permute_pts(
                    streams.reshape(-1, lengths[0], 3), type='flip')
                streams = streams_perm.reshape(-1, 3)
        else:
            # streams, lengths = load_selected_streamlines(T_file,
            #                                         sample['points'].tolist())
            streams, lengths = load_streamlines_fast(T_file,
                                                    sample['points'].tolist())
            #print('\ttime loading selected streamlines %f' % (time.time()-t0))
            if self.centering:
                streams_centered = self.center_sl_list(
                    np.split(streams, np.cumsum(lengths))[:-1])
                streams = np.vstack(streams_centered)

            if self.permute:
                streams_perm = self.permute_pts(
                    np.split(streams, np.cumsum(lengths))[:-1], type='flip')
                streams = streams_perm.reshape(-1, 3)
        
        if self.standardize:
            stats_file = glob.glob(sub_dir + '/*_stats.npy')[0]
            mu, sigma, M, m = np.load(stats_file)
            streams = (streams - mu) / sigma



        #t0 = time.time()
        sample['points'] = self.build_graph_sample(streams,
                    lengths,
                    torch.from_numpy(sample['gt']) if self.with_gt else None)
        #sample['tract'] = streamlines
        #print('sample:',sample['points'])
        #print('\ttime building graph %f' % (time.time()-t0))
        return sample

    def center_sl_list(self, sl_list):
        centers = np.array(map(functools.partial(np.mean, axis=0), sl_list))
        return map(np.subtract, sl_list, centers)

    def permute_pts(self, sl_list, type='rand'):
        perm_sl_list = []
        for sl in sl_list:
            if type == 'flip':
                perm_sl_list.append(sl[::-1])
            else:
                perm_idx = torch.randperm(len(sl)).tolist()
                perm_sl_list.append(sl[perm_idx])
        return np.array(perm_sl_list)


    def build_graph_sample(self, streams, lengths, gt=None):
        #t0 = time.time()
        #print('time numpy split %f' % (time.time()-t0))
        ### create graph structure
        #sls_lengths = torch.from_numpy(sls_lengths)
        lengths = torch.from_numpy(lengths).long()
        #print('sls lengths:',sls_lengths)
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             #sls_lengths=sls_lengths,
                             bvec=batch_vec,
                             pos=streams)
        #                     bslices=batch_slices)
        #edges = torch.empty((2, 2*l - 2*n), dtype=torch.long)
        if self.return_edges:
            e1 = set(np.arange(0,l-1)) - set(slices.numpy()-1)
            e2 = set(np.arange(1,l)) - set(slices.numpy())
            edges = torch.tensor([list(e1)+list(e2),list(e2)+list(e1)],
                            dtype=torch.long)
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges,1)
            graph_sample['edge_attr'] = edge_attr
        if self.distance:
            graph_sample = self.distance(graph_sample)
        #if self.self_loops:
        #graph_sample = self.self_loops(graph_sample)
        if gt is not None:
            graph_sample['y'] = gt

        return graph_sample
