import numpy as np
import cffi
import subprocess
import os
import glob

root = os.path.dirname(os.path.dirname(__file__))

libs = glob.glob(root + "/fast_sweeping_capi.*.so")

if len(libs) == 0:
    raise(OSError("fast_sweeping_capi.*.so library not found"))

if len(libs) > 1:
    raise(OSError(("More then one candidate of fast_sweeping_capi.*.so library found: {}").format(libs)))

ffi = cffi.FFI()
lib = ffi.dlopen(libs[0])

ffi.cdef("""
void signed_distance_2d(double*, double*, size_t, size_t, double);
void signed_distance_3d(double*, double*, size_t, size_t, size_t, double);
double hausdorff_dist_2d(double*, double*, size_t, size_t, double);
double l2_hausdorff_dist_2d(double*, double*, size_t, size_t, double);
double hausdorff_dist_3d(double*, double*, size_t, size_t, size_t, double);
double l2_hausdorff_dist_3d(double*, double*, size_t, size_t, size_t, double);

""")

def verify_ffi_array(u, dtype):
    if u.dtype != dtype:
        raise TypeError('Array must be of type {}.'.format(dtype))
    if not u.flags.c_contiguous:
        raise TypeError('Array must be C contiguos.')
    if not u.flags.aligned:
        raise TypeError('Array must be properly aligned.')

def signed_distance(u, h, out = None):
    """
    Computes the signed distance to the zero level set of the function `u`
    given on a regular grid with spacing `h`.

    You can use `out` to specify an array where to store the result.
    """
    verify_ffi_array(u, np.dtype('float64'))

    if out is None:
        d = np.zeros_like(u, order='c')
    else:
        d = out

    verify_ffi_array(d, np.dtype('float64'))

    pd = ffi.cast("double *", d.ctypes.data)
    pu = ffi.cast("double *", u.ctypes.data)

    if len(u.shape) == 2:
        i, j = u.shape

        lib.signed_distance_2d(pd, pu, i, j, h)
    elif len(u.shape) == 3:
        i, j, k = u.shape
        lib.signed_distance_3d(pd, pu, i, j, k, h)
    else:
        raise TypeError("Array must be 2 or 3 dimensional.")

    return d

def hausdorff_dist(u, v, h, l2=True):
    """ Returns the Hausdorff distance between two zero level sets.

    Returns +∞ if at least one of the level sets is empty."""
    verify_ffi_array(u, np.dtype('float64'))
    verify_ffi_array(v, np.dtype('float64'))

    if u.shape != v.shape:
        raise TypeError("Arrays must have the same shape.")

    pu = ffi.cast("double *", u.ctypes.data)
    pv = ffi.cast("double *", v.ctypes.data)

    if len(u.shape) == 2:
        i, j = u.shape

        if l2:
            return lib.l2_hausdorff_dist_2d(pu, pv, i, j, h)
        else:
            return lib.hausdorff_dist_2d(pu, pv, i, j, h)
    elif len(u.shape) == 3:
        i, j, k = u.shape

        if l2:
            return lib.l2_hausdorff_dist_3d(pu, pv, i, j, k, h)
        else:
            return lib.hausdorff_dist_3d(pu, pv, i, j, k, h)
    else:
        raise TypeError("Array must be 2 or 3 dimensional.")
