extern crate fast_sweeping;
extern crate libc;

use libc::size_t;
use std::slice;

#[macro_use]
mod macros;

sign_dist_ffi_fn! {
    fn signed_distance_2d(ni, nj) -> f64
}

sign_dist_ffi_fn! {
    fn signed_distance_3d(ni, nj, nk) -> f64
}

haus_dist_ffi_fn! {
    fn hausdorff_dist_2d(ni, nj) -> f64
}

haus_dist_ffi_fn! {
    fn l2_hausdorff_dist_2d(ni, nj) -> f64
}

haus_dist_ffi_fn! {
    fn hausdorff_dist_3d(ni, nj, nk) -> f64
}

haus_dist_ffi_fn! {
    fn l2_hausdorff_dist_3d(ni, nj, nk) -> f64
}
