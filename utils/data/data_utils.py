from dipy.segment.clustering import qbx_and_merge
import datasets as ds
import nibabel as nib
import torch
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points, length, Streamlines
from dipy.align.streamlinear import DEFAULT_BOUNDS, StreamlineLinearRegistration, progressive_slr, remove_clusters_by_size
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.trk import Field
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import Batch as gBatch
from torch_geometric.data import DataListLoader as gDataLoader
from torchvision import transforms

from .transforms import *


def get_dataset(cfg, trans, train=True):
    if not train:
        sub_list = cfg['sub_list_val']
        batch_size = 1
        shuffling = False
        run = 'val'
    else:
        sub_list = cfg['sub_list_train']
        batch_size = int(cfg['batch_size'])
        shuffling = cfg['shuffling']
        run = 'train'

    if 'bids' in cfg['dataset']:
        dataset = ds.BIDSDataset(sub_list,
                                  cfg['dataset_dir'],
                                  run=run,
                                  data_name=cfg['data_name'],
                                  transform=transforms.Compose(trans),
                                  return_edges=cfg['return_edges'],
                                  labels_dir=cfg['labels_dir'],
                                  labels_name=cfg['labels_name'],
                                  with_gt=True,)
    if 'hcp20' in cfg['dataset']:
        dataset = ds.HCP20Dataset(sub_list,
                                  cfg['dataset_dir'],
                                  same_size=cfg['same_size'],
                                  transform=transforms.Compose(trans),
                                  return_edges=cfg['return_edges'],
                                  load_one_full_subj=False,
                                  labels_dir=cfg['labels_dir'])
    if cfg['dataset'] == 'atlasparts':
        dataset = ds.StreamAtlasDataset(sub_list,
                                        cfg['dataset_dir'],
                                        data_name=cfg['data_name'],
                                        run=run,
                                        labels_dir=cfg['labels_dir'],
                                        lbls_name=cfg['labels_name'],
                                        same_size=cfg['same_size'],
                                        transform=transforms.Compose(trans),
                                        with_gt=True,
                                        return_edges=cfg['return_edges'],
                                        self_loops=False,
                                        data_per_point=False,
                                        add_tangent=False)

    dataloader = gDataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffling,
                             num_workers=int(cfg['n_workers']),
                             pin_memory=True)

    print("Dataset %s loaded, found %d samples" %
          (cfg['dataset'], len(dataset)))
    return dataset, dataloader


def get_transforms(cfg, train=True):
    trans = []

    if cfg['rnd_sampling']:
        if train:
            trans.append(RndSampling(cfg['fixed_size'], maintain_prop=False),
                        prop_vector=cfg['sampling_prop_vector'])
        else:
            trans.append(FixedRndSampling(cfg['fixed_size']))

    print(trans)
    return trans


def get_gbatch_sample(sample, sample_size, same_size, return_name=False):
    data_list = []
    name_list = []
    ori_batch = []
    for i, d in enumerate(sample):
        if 'bvec' in d['points'].keys:
            d['points'].bvec += sample_size * i
        data_list.append(d['points'])
        name_list.append(d['name'])
        ori_batch.append([i] * sample_size)
    points = gBatch().from_data_list(data_list)
    points.ori_batch = torch.tensor(ori_batch).flatten().long()
    if 'bvec' in points.keys:
        #points.batch = points.bvec.copy()
        points.batch = points.bvec.clone()
        del points.bvec
    if same_size:
        points['lengths'] = points['lengths'][0].item()

    if return_name:
        return points, name_list
    return points


def resample_streamlines(streamlines, n_pts=16):
    resampled = []
    for sl in streamlines:
        resampled.append(set_number_of_points(sl, n_pts))

    return resampled


def tck2trk(tck_fn, nii_fn, out_fn=None):
    nii = nib.load(nii_fn)
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(tck_fn)
    if out_fn is None:
        out_fn = tck_fn[:-4] + '.trk'
    nib.streamlines.save(tck.tractogram, out_fn, header=header)


def trk2tck(trk_fn, out_fn=None):
    trk = nib.streamlines.load(trk_fn)
    if out_fn is None:
        out_fn = trk_fn[:-4] + '.tck'
    nib.streamlines.save(trk.tractogram, out_fn)


def slr_with_qbx_partial(centroids_static,
                 moving,
                 x0='affine',
                 rm_small_clusters=50,
                 maxiter=100,
                 select_random=None,
                 verbose=False,
                 greater_than=50,
                 less_than=250,
                 qbx_thr=[40, 30, 20, 15],
                 nb_pts=20,
                 progressive=True,
                 rng=None,
                 num_threads=None):
    """ See dipy api
    """
    if rng is None:
        rng = np.random.RandomState()

    if verbose:
        print('Moving streamlines size {}'.format(len(moving)))

    def check_range(streamline, gt=greater_than, lt=less_than):

        if (length(streamline) > gt) & (length(streamline) < lt):
            return True
        else:
            return False

    streamlines2 = Streamlines(moving[np.array(
        [check_range(s) for s in moving])])
    if verbose:
        print('Moving streamlines after length reduction {}'.format(
            len(streamlines2)))

    qb_centroids1 = Streamlines(centroids_static)
    qb_centroids1._data.astype('f4')

    if select_random is not None:
        rstreamlines2 = select_random_set_of_streamlines(streamlines2,
                                                         select_random,
                                                         rng=rng)
    else:
        rstreamlines2 = streamlines2

    rstreamlines2 = set_number_of_points(rstreamlines2, nb_pts)
    rstreamlines2._data.astype('f4')

    cluster_map2 = qbx_and_merge(rstreamlines2, thresholds=qbx_thr, rng=rng)

    qb_centroids2 = remove_clusters_by_size(cluster_map2, rm_small_clusters)

    if not progressive:
        slr = StreamlineLinearRegistration(x0=x0,
                                           options={'maxiter': maxiter},
                                           num_threads=num_threads)
        slm = slr.optimize(qb_centroids1, qb_centroids2)
    else:
        bounds = DEFAULT_BOUNDS

        slm = progressive_slr(qb_centroids1,
                              qb_centroids2,
                              x0=x0,
                              metric=None,
                              bounds=bounds,
                              num_threads=num_threads)

    if verbose:
        print('QB static centroids size %d' % len(qb_centroids1, ))
        print('QB moving centroids size %d' % len(qb_centroids2, ))

    moved = slm.transform(moving)

    return moved, slm.matrix, qb_centroids2
