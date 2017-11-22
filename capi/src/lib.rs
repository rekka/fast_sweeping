extern crate fast_sweeping;
extern crate libc;

use libc::size_t;
use std::slice;

#[macro_use]
mod macros;


ffi_fn! {
    fn signed_distance_2d(d: *mut f64, u: *const f64, ni: size_t, nj: size_t, h: f64) {
        let ni = ni as usize;
        let nj = nj as usize;
        let len = ni * nj;

        let d = unsafe { slice::from_raw_parts_mut(d, len) };
        let u = unsafe { slice::from_raw_parts(u, len) };

        fast_sweeping::signed_distance_2d(d, u, (ni, nj), h);
    }
}

dist_ffi_fn! {
    fn hausdorff_dist_2d(ni, nj) -> f64
}

dist_ffi_fn! {
    fn l2_hausdorff_dist_2d(ni, nj) -> f64
}

dist_ffi_fn! {
    fn hausdorff_dist_3d(ni, nj, nk) -> f64
}

dist_ffi_fn! {
    fn l2_hausdorff_dist_3d(ni, nj, nk) -> f64
}
