from struct import unpack
from time import time

import nibabel as nib
import os
import struct
import numpy as np
from dipy.tracking.metrics import frenet_serret, magn
from nibabel.affines import apply_affine
from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm


def get_length_struct(f, nb_bytes_int32=4, int32_fmt='<i'):
    """Parse an int32 from a file. struct.unpack() version.
    """
    return unpack(int32_fmt, f.read(nb_bytes_int32))[0]


def load_selected_streamlines(trk_fn,
                              idxs=None,
                              return_scalars=False,
                              compute_tan=False,
                              rescale=False):

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    vol_size = header['dimensions']
    
    if idxs is None:
        idxs = np.arange(nb_streamlines)
        
    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    get_length = get_length_struct

    lengths = np.empty(nb_streamlines, dtype=np.int)

    with open(trk_fn, 'rb') as f:
        f.seek(header_size)
        for idx in range(nb_streamlines):
            l = get_length(f)
            lengths[idx] = l
            jump = point_bytes * l + properties_bytes
            f.seek(jump, 1)

    # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size

    n_floats = lengths * point_size  # better because it skips properties, if they exist
    streams = np.empty((lengths[idxs].sum(), 3), dtype=np.float32)
    scalars = np.empty(
        (lengths[idxs].sum(),
         n_scalars), dtype=np.float32) if n_scalars > 0 else None
    tan_arr = np.empty((lengths[idxs].sum(), 3), dtype=np.float32)
    j = 0
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)

            if n_scalars > 0:
                scalars[j:j+lengths[idx], :] = s[:, 3:]
                s = s[:, :3]
            if compute_tan:
                tan_arr[j:j + lengths[idx], :] = abs(tan_dipy(s))

            streams[j:j+lengths[idx], :] = s
            j += lengths[idx]

    # rescale
    if rescale:
        streams /= vol_size

    # apply affine
    aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
    streams = apply_affine(aff, streams)
   
    if return_scalars:
        streams = np.hstack((streams, scalars))

    if compute_tan:
        streams = np.hstack((streams, tan_arr))

    return streams, lengths[idxs]


def load_selected_streamlines_uniform_size(trk_fn,
                                           idxs=None,
                                           return_scalars=False,
                                           compute_tan=False,
                                           rescale=False):

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    vol_size = header['dimensions']
    if idxs is None:
        idxs = np.arange(nb_streamlines)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    get_length = get_length_struct

    with open(trk_fn, 'rb') as f:
        f.seek(header_size)
        l = get_length(f)

    lengths = np.array([l] * nb_streamlines).astype(np.int)

     # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size

    n_floats = lengths * point_size  # better because it skips properties, if they exist
    streams = np.empty((lengths[idxs].sum(), 3), dtype=np.float32)
    scalars = np.empty(
        (lengths[idxs].sum(),
         n_scalars), dtype=np.float32) if n_scalars > 0 else None
    tan_arr = np.empty(
        (lengths[idxs].sum(), 3), dtype=np.float32) if compute_tan else None
    j = 0
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)

            if n_scalars > 0:
                scalars[j:j+lengths[idx], :] = s[:, 3:]
                s = s[:, :3]

            if compute_tan:
                # use abs to avoid orientation problem
                tan_arr[j:j + lengths[idx], :] = abs(tan_dipy(s))

            streams[j:j + lengths[idx], :] = s
            j += lengths[idx]
    # rescale
    if rescale:
        streams /= vol_size

    # apply affine
    aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
    streams = apply_affine(aff, streams)

    if return_scalars:
        streams = np.hstack((streams, scalars))

    if compute_tan:
        streams = np.hstack((streams, tan_arr))

    return streams, lengths[idxs]


def tan_dipy(xyz):
    # taken from dipy.tracking.metrics.frenet_serret
    n_pts = xyz.shape[0]
    if n_pts == 0:
        raise ValueError('xyz array cannot be empty')

    dxyz = np.gradient(xyz)[0]
    # Tangent
    T = np.divide(dxyz, magn(dxyz, 3))
    return T


def load_selected_streamlines_uniform_size_seq(trk_fn,
                                               idxs=None,
                                               return_scalars=False,
                                               sequential=False):

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    if idxs is None:
        idxs = np.arange(nb_streamlines)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    with open(trk_fn, 'rb') as f:
        f.seek(header_size)
        l = np.fromfile(f, np.int32, 1)[0]

    lengths = np.array([l] * len(idxs)).astype(np.int)

    # float for each streamline, the last + 1 is for the length float
    n_floats = l * point_size + n_properties + 1

    streams = np.empty((l * len(idxs), 3), dtype=np.float32)
    scalars = np.empty(
        (l * len(idxs),
         n_scalars), dtype=np.float32) if n_scalars > 0 else None
    j = 0
    idx0 = idxs[0]
    with open(trk_fn, 'rb') as f:
        idx_bytes = (l * point_bytes + properties_bytes + length_bytes) * idx0
        if sequential:
            # import ipdb; ipdb.set_trace()
            f.seek(header_size + idx_bytes)
            s = np.fromfile(f, np.float32, n_floats * len(idxs))
            s.resize(len(idxs), n_floats)
            # remove length and properties
            s = s[:, :-(n_properties + 1)]
            # s = s.resize()
            s = s.reshape(l * len(idxs), point_size)

            if n_scalars > 0:
                scalars = s[:, 3:]
                s = s[:, :3]

            streams = s

        else:
            for idx in idxs:
                idx_bytes = (l * point_bytes + properties_bytes +
                             length_bytes) * idx0
                # move to the position initial position of the coordinates
                # of the streamline:
                f.seek(idx_bytes)
                # Parse the floats:
                s = np.fromfile(f, np.float32, n_floats)
                s.resize(l, point_size)

                if n_scalars > 0:
                    scalars[j:j + l, :] = s[:, 3:]
                    s = s[:, :3]

                streams[j:j + l, :] = s
                j += l
    # apply affine
    aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
    streams = apply_affine(aff, streams)

    if return_scalars:
        streams = np.hstack((streams, scalars))

    return streams, lengths


def fast_load_streamlines(trk_fn):
    streams, lengths = load_selected_streamlines(trk_fn)
    streamlines = np.split(streams, np.cumsum(lengths[:-1]))
    return np.array(streamlines, dtype=np.object)