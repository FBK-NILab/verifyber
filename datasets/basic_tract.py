import functools
import nibabel as nib
import numpy as np
import torch
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset

from utils.data.selective_loader_numba import load_streamlines as load_streamlines_fast


class TractDataset(gDataset):
    def __init__(self,
                 T_file,
                 transform=None,
                 return_edges=True,
                 split_obj=True,
                 seq_load=False,
                 in_memory=True):
        self.T_file = T_file
        self.transform = transform
        self.return_edges = return_edges
        self.remaining = [[]]
        self.split_obj = split_obj
        self.seq_load = seq_load
        self.load_sel_streamlines = functools.partial(
            load_streamlines_fast, container='array_flat'
        )
        self.in_memory = in_memory

        if in_memory:
            print('loading all trk to memory...')
            import time
            t0 = time.time()
            T, L = self.load_sel_streamlines(T_file)
            print(f'tractogram loaded in {time.time() - t0}sec')

            self.L_mem = L
            self.T_mem = np.array(np.split(T, L.cumsum()[:-1], axis=0),
                             dtype=np.object)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item

    def getitem(self, idx):
        T = nib.streamlines.load(self.T_file, lazy_load=True)

        if len(self.remaining[idx]) == 0:
            self.remaining[idx] = set(np.arange(T.header['nb_streamlines']))

        sample = {'points': np.array(list(self.remaining[idx]))}

        if self.transform:
            sample = self.transform(sample)
        if self.split_obj:
            self.remaining[idx] -= set(sample['points'])
            sample['obj_idxs'] = sample['points'].copy()
            sample['obj_full_size'] = T.header['nb_streamlines']
        n = len(sample['points'])
        idxs_to_load = sample['points'].tolist()

        if self.in_memory:
            streams = np.concatenate(self.T_mem[idxs_to_load], axis=0).astype(np.float32)
            lengths = self.L_mem[idxs_to_load]
        else:
            streams, lengths = self.load_sel_streamlines(
                self.T_file, idxs_to_load)
        sample['points'] = self.build_graph_sample(streams, lengths, gt=None)
        return sample

    def build_graph_sample(self, streams, lengths, gt=None):
        lengths = torch.from_numpy(lengths).long()
        batch_vec = torch.arange(len(lengths)).repeat_interleave(lengths)
        batch_slices = torch.cat([torch.tensor([0]), lengths.cumsum(dim=0)])
        slices = batch_slices[1:-1]
        streams = torch.from_numpy(streams)
        l = streams.shape[0]
        graph_sample = gData(x=streams,
                             lengths=lengths,
                             bvec=batch_vec,
                             pos=streams)
        e1 = set(np.arange(0, l - 1)) - set(slices.numpy() - 1)
        e2 = set(np.arange(1, l)) - set(slices.numpy())
        edges = torch.tensor(
            [list(e1) + list(e2), list(e2) + list(e1)], dtype=torch.long)
        graph_sample['edge_index'] = edges
        num_edges = graph_sample.num_edges
        edge_attr = torch.ones(num_edges, 1)
        graph_sample['edge_attr'] = edge_attr
        return graph_sample
