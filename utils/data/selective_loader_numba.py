# Copyright (c) 2019 Pietro Astolfi, Emanuele Olivetti
# MIT License

import numpy as np
import nibabel as nib
from numba import jit
import os
from time import time
from dipy.tracking.streamlinespeed import length, set_number_of_points


@jit(nopython=True)
def parse_lengths(buffer, lengths, point_size, n_properties):
    pointer = 0
    for idx in range(lengths.size):
        l = buffer[pointer]
        lengths[idx] = l
        pointer += 1 + l * point_size + n_properties

    return lengths


@jit(nopython=True)
def parse_streamlines(buffer,
                      idxs,
                      split_points,
                      n_floats,
                      affine,
                      apply_affine=True,
                      rescale_factor=None,
                      reorient=False):
    streamlines = []
    rotation_zoom_shear = affine[:3, :3].copy(
    )  # necessary to create a C-contiguous array
    translation = affine[:3, 3:4].copy(
    )  # necessary to create a C-contiguous array
    for idx in idxs:  # range(n_floats.size):
        s = buffer[split_points[idx]:split_points[idx] +
                   n_floats[idx]].reshape(-1, 3)
        if apply_affine:
            s = (np.dot(rotation_zoom_shear, s.T) + translation).T
        if rescale_factor is not None:
            s *= rescale_factor
        if reorient:
            o = np.zeros(3) # origin
            start = np.argmin(np.array([np.sqrt(np.sum((o - s[0])**2)),
                np.sqrt(np.sum((o - s[-1])**2))]))
            if start == 1:
                s = s[::-1]

        streamlines.append(s)

    return streamlines


def load_streamlines(trk_fn,
                     idxs=None,
                     apply_affine=True,
                     resample=False,
                     container='array_flat',
                     replace=False,
                     verbose=False,
                     load_twice=True,
                     rescale=False,
                     reorient=False,
                     return_len=False):
    """Load streamlines from a .trk file. If a list of indices (idxs) is
    given, this function just loads and returns the requested
    streamlines, skipping all the non-requested ones.

    This function is sort of similar to nibabel.streamlines.load() but
    extremely FASTER. It is very convenient if you need to load only
    some streamlines in large tractograms. Like 100x faster than what
    you can get with nibabel.
    """

    if verbose:
        print("Loading %s" % trk_fn)

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    aff = nib.streamlines.trk.get_affine_trackvis_to_rasmm(header)
    vol_size = header['dimensions']
    vox_size = header['voxel_sizes']

    if idxs is None:
        idxs = np.arange(nb_streamlines, dtype=np.int32)
    elif isinstance(idxs, int):
        if verbose:
            print('Sampling %s streamlines uniformly at random' % idxs)

        if idxs > nb_streamlines and (not replace):
            print('WARNING: Sampling with replacement')

        idxs = np.random.choice(np.arange(nb_streamlines),
                                idxs,
                                replace=replace)
    elif isinstance(idxs, list):
        idxs = np.array(idxs, dtype=np.int)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    nb_bytes_float32 = np.dtype('<f').itemsize

    if verbose:
        print("Loading the whole data blob.")
        t0 = time()

    buffer = np.empty((os.path.getsize(trk_fn) // nb_bytes_float32),
                      np.float32)
    with open(trk_fn, 'rb') as f:
        f.seek(1000)  # 1000 is the size of the header in bytes
        f.readinto(buffer)

    if verbose:
        print("%s values read in %s sec." % (buffer.size, time() - t0))

    if verbose:
        print("Parsing lengths of %s streamlines" % nb_streamlines)
        t0 = time()

    lengths = np.empty(nb_streamlines, dtype=np.int32)
    buffer_int32 = buffer.view(dtype=np.int32)
    lengths = parse_lengths(buffer_int32, lengths, point_size, n_properties)

    if verbose:
        print("%s sec." % (time() - t0))

    n_floats = lengths * point_size
    split_points = (n_floats + 1 +
                    n_properties).cumsum() - n_floats - n_properties

    if verbose:
        print("Extracting %s streamlines" % nb_streamlines)
        if apply_affine:
            print("and applying the affine")

        t0 = time()

    scaling = 1/(vol_size * vox_size) if rescale else None

    streamlines = parse_streamlines(buffer, idxs, split_points, n_floats, aff,
                                    apply_affine, scaling, reorient)

    if verbose:
        print("%s sec." % (time() - t0))

    if verbose:
        print("Converting all streamlines to the container %s" % container)
        t0 = time()

    if resample == 'fixed_step':
        for i, s in enumerate(streamlines):
            l = length(s)
            lengths[idxs[i]] = int(l/5)
            streamlines[i] = set_number_of_points(s, lengths[idxs[i]])
    elif resample:
        streamlines = set_number_of_points(streamlines, resample)
        lengths[:] = resample

    if container == 'array':
        streamlines = np.array(streamlines, dtype=np.object)
    elif container == 'ArraySequence':
        streamlines = nib.streamlines.ArraySequence(streamlines)
    elif container == 'list':
        pass
    elif container == 'array_flat':
        streamlines = np.concatenate(streamlines, axis=0)
        return_len = True
    else:
        raise Exception

    if verbose:
        print("%s sec." % (time() - t0))

    if return_len:
        return streamlines, lengths[idxs]
    else:
        return streamlines


if __name__ == '__main__':

    np.random.seed(0)

    trk_fn = 'sub-100206_var-FNAL_tract.trk'
    trk_fn = 'sub-599469_var-10M_tract.trk'

    # idxs = np.random.choice(500000, 200000, replace=True)
    # idxs.sort()
    idxs = None  # This is for loading all streamlines
    # idxs = 1000

    streamlines, header, lengths, idxs = load_streamlines(trk_fn,
                                                          idxs,
                                                          apply_affine=True,
                                                          container='list',
                                                          verbose=True)

    print("Done.")
