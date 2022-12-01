import glob

import numpy as np
import torch

class RndSampling(object):
    """Random sampling from input object to return a fixed size input object
    Args:
        output_size (int): Desired output size.
        maintain_prop (bool): Default True. Indicates if the random sampling
            must be proportional to the number of examples of each class
    """
    def __init__(self, output_size, maintain_prop=False, prop_vector=[]):
        assert isinstance(output_size, (int))
        assert isinstance(maintain_prop, (bool))
        self.output_size = output_size
        self.maintain_prop = maintain_prop
        self.prop_vector = prop_vector

    def __call__(self, sample):
        np.random.seed(np.uint32(torch.initial_seed()))

        pts = sample['points']
        gt = sample['gt'] if 'gt' in sample else None

        n = pts.shape[0]
        if self.maintain_prop:
            n_classes = gt.max() + 1
            remaining = self.output_size
            chosen_idx = []
            for cl in reversed(range(n_classes)):
                if (gt == cl).sum() == 0:
                    continue
                if cl == gt.min():
                    chosen_idx += np.random.choice(
                        np.argwhere(gt == cl).reshape(-1),
                        int(remaining)).reshape(-1).tolist()
                    break
                prop = float(np.sum(gt == cl)) / n
                k = np.round(self.output_size * prop)
                remaining -= k
                chosen_idx += np.random.choice(
                    np.argwhere(gt == cl).reshape(-1),
                    int(k)).reshape(-1).tolist()

            assert (self.output_size == len(chosen_idx))
            chosen_idx = np.array(chosen_idx)
        elif len(self.prop_vector) != 0:
            n_classes = gt.max() + 1
            while len(self.prop_vector) < n_classes:
                self.prop_vector.append(1)
            remaining = self.output_size
            out_size = self.output_size
            chosen_idx = []
            excluded = 0
            for cl in range(n_classes):
                if (gt == cl).sum() == 0:
                    continue
                if cl == gt.max():
                    chosen_idx += np.random.choice(
                        np.argwhere(gt == cl).reshape(-1),
                        int(remaining)).reshape(-1).tolist()
                    break
                if self.prop_vector[cl] != 1:
                    prop = self.prop_vector[cl]
                    excluded += np.sum(gt == cl)
                    #excluded -= (n-excluded)*prop
                    k = np.round(self.output_size * prop)
                    out_size = remaining - k
                else:
                    prop = float(np.sum(gt == cl)) / (n - excluded)
                    k = np.round(out_size * prop)

                remaining -= k
                chosen_idx += np.random.choice(
                    np.argwhere(gt == cl).reshape(-1),
                    int(k)).reshape(-1).tolist()

            assert (self.output_size == len(chosen_idx))
            chosen_idx = np.array(chosen_idx)
        else:
            chosen_idx = np.random.choice(range(n), self.output_size)

        out_gt = gt[chosen_idx] if isinstance(gt, (list, np.ndarray)) and len(gt) > 1 else gt
        return {'points': pts[chosen_idx], 'gt': out_gt}


class FixedRndSampling(object):
    """Fixed random sampling from input object to return a fixed size input object

    Args:
        output_size (int): Desired output size.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        pts = sample['points']
        gt = sample['gt'] if 'gt' in sample else None
        n = pts.shape[0]
        out_n = self.output_size

        idxs = range(n)
        step = n / out_n
        chosen_idx = idxs[::step]

        if len(chosen_idx) < out_n:
            remaining = out_n - len(chosen_idx)
            del idxs[::step]
            chosen_idx.append(idxs[:remaining])

        pts = pts[chosen_idx]
        gt = gt[chosen_idx] if isinstance(gt, (list, np.ndarray)) and len(gt) > 1 else gt

        return {'points': pts, 'gt': gt}


class TestSampling(object):
    """Random sampling from input object until the object is all sampled
    Args:
        output_size (int): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        sl = sample['points']

        n = sl.shape[0]
        if self.output_size > len(range(n)):
            chosen_idx = range(n)
        else:
            chosen_idx = np.random.choice(range(n),
                                          self.output_size,
                                          replace=False).tolist()
        out_sample = {'points': sl[chosen_idx]}

        if 'gt' in sample.keys():
            out_sample['gt'] = sample['gt'][chosen_idx]
        return out_sample


class SeqSampling(object):
    """Random sampling from input object until the object is all sampled
    Args:
        output_size (int): Desired output size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        sl = sample['points']

        n = sl.shape[0]
        if self.output_size > len(range(n)):
            chosen_idx = range(n)
        else:
            chosen_idx = range(self.output_size)

        out_sample = {'points': sl[chosen_idx]}

        if 'gt' in sample.keys():
            out_sample['gt'] = sample['gt'][chosen_idx]
        return out_sample


class SampleStandardization(object):
    """Standardize the sample by substracting the mean and dividing by the stddev
    """
    def __call__(self, sample):
        sl, gt = sample['points'], sample['gt']
        stats_file = glob.glob(self.sub_dir + '/*_stats.npy')[0]
        mu, sigma, M, m = np.load(stats_file)

        standardized_data = (sl - mu) / sigma

        assert (normalized_data.shape == sl.shape)

        return {'points': normalized_data, 'gt': gt}


class PointCloudNormalization(object):
    def __call__(self, sample):
        pts, gt = sample['points']

        #get furthest point distance then normalize
        d = max(np.sum(np.abs(pts)**2, axis=-1)**(1. / 2))
        pts /= d

        sample['points'] = pts
        return sample


class PointCloudCentering(object):
    def __call__(self, sample):
        pts, gt = sample['points'], sample['gt']

        d = np.mean(pts, axis=0)
        pts[:, 0] -= d[0]
        pts[:, 1] -= d[1]
        pts[:, 2] -= d[2]

        sample['points'] = pts
        return sample