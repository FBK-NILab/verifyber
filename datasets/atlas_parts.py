import glob
import os

import nibabel as nib
import numpy as np
import torch
# from torch_geometric.data import Batch as gBatch
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.utils import add_self_loops
from utils.data.selective_loader import (load_selected_streamlines,
                                         load_selected_streamlines_uniform_size
                                         )
from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast

class StreamAtlasDataset(gDataset):
    def __init__(self,
                 sub_list,
                 root_dir,
                 data_name='_atlas_merged',
                 run='train',
                 labels_dir=None,
                 lbls_name='_atlas_merged',
                 same_size=False,
                 transform=None,
                 self_loops=False,
                 with_gt=True,
                 return_edges=False,
                 split_obj=False,
                 data_per_point=False,
                 add_tangent=False):
        """
        Args:
            parts_file (string): txt list of parts to consider.
            root_dir (string): root directory of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # super(StreamAtlasDataset, self).__init__(same_size=same_size,
        #                                          self_loops=self_loops,
        #                                          return_edges=return_edges)

        self.self_loops = self_loops
        self.return_edges = return_edges
        # if same_size:
        #     self.load_streamlines = load_selected_streamlines_uniform_size
        # else:
        #     self.load_streamlines = load_selected_streamlines
        self.load_streamlines = load_streamlines_fast

        # parts_file = parts_file + '_{}.txt'.format(run)
        parts_file = sub_list

        with open(parts_file) as f:
            parts = [s.strip() for s in f.readlines()]
        self.parts = parts

        #t0 = time.time()
        self.part_dir = []
        self.trks = []
        self.fns = []
        for p in parts:
            self.lazy_load_trk(root_dir, p, data_name)
        #print('\tload lazy T %f' % (time.time()-t0))

        self.transform = transform
        self.with_gt = with_gt
        self.data_per_point = data_per_point
        self.add_tangent = add_tangent

        if run == 'train':
            split_obj = False
        if split_obj:
            self.remaining = [[] for _ in range(len(parts))]
        self.split_obj = split_obj

        #t0 = time.time()
        if with_gt:
            self.labels = []
            self.lbls_name = lbls_name
            for p in parts:
                labels_dir = root_dir if labels_dir is None else labels_dir
                self.load_labels(labels_dir, p, lbls_name)
        #print('\tload gt %f' % (time.time()-t0))
        print(self.__dict__)

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx):

        T_fn = self.fns[idx]
        T = self.trks[idx]
        gt = self.labels[idx] if self.with_gt else None

        if self.split_obj:
            if len(self.remaining[idx]) == 0:
                self.remaining[idx] = set(np.arange(
                    T.header['nb_streamlines']))
            pts = np.array(list(self.remaining[idx]))
            gt = gt[list(self.remaining[idx])] if self.with_gt else None
        else:
            pts = np.arange(T.header['nb_streamlines'])

        sample = self.init_sample(pts, gt)

        #t0 = time.time()
        if self.transform:
            sample = self.transform(sample)
        #print('\ttime sampling %f' % (time.time()-t0))

        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']

        sample['name'] = T_fn.split('/')[-1].rsplit('.', 1)[0]
        sample['dir'] = self.part_dir[idx]

        #t0 = time.time()
        n = len(sample['points'])

        streams, lengths = self.load_streamlines(
            T_fn,
            sample['points'].tolist())
        #print('\ttime loading selected streamlines %f' % (time.time()-t0))

        #t0 = time.time()
        sample['points'] = self.build_graph_sample(
            streams, lengths,
            torch.from_numpy(sample['gt']) if self.with_gt else None)
        #print('\ttime building graph %f' % (time.time()-t0))

        return sample

    def init_sample(self, pts, gt=None):
        if gt is None:
            return {'points': pts}

        return {'points': pts, 'gt': gt}

    def lazy_load_trk(self, root_dir, part, part_name):
        part_dir = os.path.join(root_dir, f'part-{part}')
        T_fn = glob.glob(os.path.join(part_dir,
                                      f'p-{part}*{part_name}*.trk'))[0]
        self.part_dir.append(part_dir)
        self.trks.append(nib.streamlines.load(T_fn, lazy_load=True))
        self.fns.append(T_fn)

    def load_labels(self, root_dir, part, lbls_name):
        part_dir = os.path.join(root_dir, f'part-{part}')
        label_fn = glob.glob(
            os.path.join(part_dir, f'p-{part}*{lbls_name}*.npy'))[0]

        if label_fn[-4:] == '.txt':
            lbl = np.loadtxt(label_fn)
            lbl[lbl == -1] = 0
            self.labels.append(lbl)
        else:
            lbl = np.load(label_fn, allow_pickle=True)
            lbl[lbl == -1] = 0
            self.labels.append(lbl)

    def permute_pts(self, sl_list):
        perm_sl_list = []
        for sl in sl_list:
            perm_idx = torch.randperm(len(sl)).tolist()
            perm_sl_list.append(sl[perm_idx])
        return np.array(perm_sl_list)

    def build_graph_sample(self, streams, lengths, gt=None):
        #t0 = time.time()
        #print('time numpy split %f' % (time.time()-t0))
        ### create graph structure
        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             bvec=batch_vec,
                             pos=streams[:, :3])

        if self.return_edges:
            e1 = set(np.arange(0, l - 1)) - set(slices.numpy() - 1)
            e2 = set(np.arange(1, l)) - set(slices.numpy())
            edges = torch.tensor(
                [list(e1) + list(e2), list(e2) + list(e1)], dtype=torch.long)
            graph_sample['edge_index'] = edges
            num_edges = graph_sample.num_edges
            edge_attr = torch.ones(num_edges, 1)
            graph_sample['edge_attr'] = edge_attr

        if self.self_loops:
            add_self_loops(graph_sample)

        if gt is not None:
            graph_sample['y'] = gt

        return graph_sample
