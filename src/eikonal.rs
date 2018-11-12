//! The fast sweeping method for the eikonal equation.
//!
//! This is an implementation of the fast sweeping method proposed by Zhao (2005). It solves the
//! eikonal equation
//!
//! ‖∇d‖ = 1,
//!
//! where ‖.‖ is a norm (even, positively homogeneous convex function, positive away from the
//! origin), by performing 2^N Gauss-Seidel sweeps in alternating directions. Here N is the
//! dimension. Zhao showed that the accuracy for the Euclidean norm is O(h log h).

use ndarray::prelude::*;
use ndarray::{azip, s};
// use ndarray_parallel::par_azip;
use std::cmp;

/// Computes the solution of the eikonal equation ‖∇d‖ = 1 in 2D using the fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
///
/// `inv_norm(d, [d1, d2], [s1, s2]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values d_i at points -s_i e_i, find the largest value t ≤ d at the origin such that
/// ‖p‖ ≤ 1, where p_i = (s_i (t - d_i))_+.
pub fn fast_sweep_2d<F>(d: &mut [f64], dim: (usize, usize), inv_norm: F)
where
    F: Fn(f64, [f64; 2], [f64; 2]) -> f64 + Sync + Send,
{
    let (ni, nj) = dim;
    let (si, _sj) = (nj, 1);
    assert_eq!(ni * nj, d.len());
    // sweep in 4 directions
    // propagate information from the corners
    for p in 1..nj {
        let s = p;
        d[s] = inv_norm(d[s], [std::f64::MAX, d[s - 1]], [1., 1.]);
        let s = (ni - 1) * si + p;
        d[s] = inv_norm(d[s], [std::f64::MAX, d[s - 1]], [-1., 1.]);
        let p = nj - 1 - p;
        let s = p;
        d[s] = inv_norm(d[s], [std::f64::MAX, d[s + 1]], [1., -1.]);
        let s = (ni - 1) * si + p;
        d[s] = inv_norm(d[s], [std::f64::MAX, d[s + 1]], [-1., -1.]);
    }
    for p in 1..ni {
        let s = p * si;
        d[s] = inv_norm(d[s], [d[s - si], std::f64::MAX], [1., 1.]);
        let s = p * si + nj - 1;
        d[s] = inv_norm(d[s], [d[s - si], std::f64::MAX], [1., -1.]);
        let p = ni - 1 - p;
        let s = p * si;
        d[s] = inv_norm(d[s], [d[s + si], std::f64::MAX], [-1., 1.]);
        let s = p * si + nj - 1;
        d[s] = inv_norm(d[s], [d[s + si], std::f64::MAX], [-1., -1.]);
    }
    // sweep in diagonal bands to take advantage of instruction-level parallelism
    // 1, 1
    for band in 1..nj {
        let len = cmp::min(band + 1, ni);
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si - 1, 1)),
            &mut d[band..][..(len - 1) * (si - 1) + 2],
        ).unwrap();
        let (input, mut output) = view.split_at(Axis(1), 1);
        let di = input.slice(s![..-1, 0]);
        let dj = input.slice(s![1.., 0]);
        let mut out = output.slice_mut(s![1.., 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [1., 1.])});
    }
    for band in 1..ni - 1 {
        let len = cmp::min(ni - band, nj);
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si - 1, 1)),
            &mut d[(band + 1) * si - 1..][..(len - 1) * (si - 1) + 2],
        ).unwrap();
        let (input, mut output) = view.split_at(Axis(1), 1);
        let di = input.slice(s![..-1, 0]);
        let dj = input.slice(s![1.., 0]);
        let mut out = output.slice_mut(s![1.., 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [1., 1.])});
    }
    // -1, -1
    for band in (1..ni - 1).rev() {
        let len = cmp::min(ni - band, nj);
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si - 1, 1)),
            &mut d[(band + 1) * si - 2..][..(len - 1) * (si - 1) + 2],
        ).unwrap();
        let (mut output, input) = view.split_at(Axis(1), 1);
        let di = input.slice(s![1.., 0]);
        let dj = input.slice(s![..-1, 0]);
        let mut out = output.slice_mut(s![..-1, 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [-1., -1.])});
    }
    for band in (1..nj).rev() {
        let len = cmp::min(band + 1, ni);
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si - 1, 1)),
            &mut d[band - 1..][..(len - 1) * (si - 1) + 2],
        ).unwrap();
        let (mut output, input) = view.split_at(Axis(1), 1);
        let di = input.slice(s![1.., 0]);
        let dj = input.slice(s![..-1, 0]);
        let mut out = output.slice_mut(s![..-1, 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [-1., -1.])});
    }
    // 1, -1
    for band in (1..nj - 1).rev() {
        let len = cmp::min(nj - band, ni);
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si + 1, 1)),
            &mut d[band - 1..][..(len - 1) * (si + 1) + 2],
        ).unwrap();
        let (mut output, input) = view.split_at(Axis(1), 1);
        let di = input.slice(s![..-1, 0]);
        let dj = input.slice(s![1.., 0]);
        let mut out = output.slice_mut(s![1.., 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [1., -1.])});
    }
    // special handling of corner in band 0
    d[si] = inv_norm(d[si], [d[0], d[si + 1]], [1., -1.]);
    for band in 0..ni - 1 {
        let len = cmp::min(ni - band, nj) - if band == 0 { 1 } else { 0 };
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si + 1, 1)),
            &mut d[band * si + if band == 0 { si + 1 } else { 0 } - 1..]
                [..(len - 1) * (si + 1) + 2],
        ).unwrap();
        let (mut output, input) = view.split_at(Axis(1), 1);
        let di = input.slice(s![..-1, 0]);
        let dj = input.slice(s![1.., 0]);
        let mut out = output.slice_mut(s![1.., 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [1., -1.])});
    }
    // -1, 1
    // need special handling of corner with coords (ni-1, nj-1) due to overflow
    for band in (0..ni - 1).rev() {
        let len = cmp::min(ni - band, nj) - if nj == ni - band {
            let s = (ni - 2) * si + nj - 1;
            d[s] = inv_norm(d[s], [d[s + si], d[s - 1]], [-1., 1.]);
            1
        } else {
            0
        };
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si + 1, 1)),
            &mut d[band * si..][..(len - 1) * (si + 1) + 2],
        ).unwrap();
        let (input, mut output) = view.split_at(Axis(1), 1);
        let di = input.slice(s![1.., 0]);
        let dj = input.slice(s![..-1, 0]);
        let mut out = output.slice_mut(s![..-1, 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [-1., 1.])});
    }
    for band in 1..nj - 1 {
        let len = cmp::min(nj - band, ni) - if ni == nj - band {
            let s = (ni - 2) * si + nj - 1;
            d[s] = inv_norm(d[s], [d[s + si], d[s - 1]], [-1., 1.]);
            1
        } else {
            0
        };
        let mut view = ArrayViewMut::from_shape(
            (len, 2).strides((si + 1, 1)),
            &mut d[band..][..(len - 1) * (si + 1) + 2],
        ).unwrap();
        let (input, mut output) = view.split_at(Axis(1), 1);
        let di = input.slice(s![1.., 0]);
        let dj = input.slice(s![..-1, 0]);
        let mut out = output.slice_mut(s![..-1, 0]);
        azip!(mut out, di, dj in { *out = inv_norm(*out, [di, dj], [-1., 1.])});
    }
}

/// Computes the solution of the eikonal equation ‖∇d‖ = 1 in 3D using the fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
///
/// `inv_norm(d, [d1, d2, d3], [s1, s2, s3]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values d_i at points -s_i e_i, find the largest value t ≤ d at the origin such that
/// ‖p‖ ≤ 1, where p_i = (s_i (t - d_i))_+.
pub fn fast_sweep_3d<F>(d: &mut [f64], dim: (usize, usize, usize), mut inv_norm: F)
where
    F: FnMut(f64, [f64; 3], [f64; 3]) -> f64,
{
    let (nx, ny, nz) = dim;
    assert_eq!(nx * ny * nz, d.len());
    let (sx, sy, _sz) = (ny * nz, nz, 1);
    // see fast_sweep_2d for discussion
    for p in 1..nx {
        for dir in 0..2 {
            let (di_plane, djk, isign) = if dir == 0 {
                let (low, high) = d.split_at_mut(p * sx);
                (&low[(p - 1) * sx..][..sx], &mut high[..sx], 1.)
            } else {
                let i = nx - 1 - p;
                let (low, high) = d.split_at_mut((i + 1) * sx);
                (&high[..sx], &mut low[i * sx..][..sx], -1.)
            };
            for q in 1..ny {
                for dir in 0..2 {
                    let (di, dj, dk, jsign) = if dir == 0 {
                        let j = q;
                        let (low, high) = djk.split_at_mut(j * sy);
                        (
                            &di_plane[j * sy..][..sy],
                            &low[(j - 1) * sy..][..sy],
                            &mut high[..sy],
                            1.,
                        )
                    } else {
                        let j = ny - 1 - q;
                        let (low, high) = djk.split_at_mut((j + 1) * sy);
                        (
                            &di_plane[j * sy..][..sy],
                            &high[..sy],
                            &mut low[j * sy..][..sy],
                            -1.,
                        )
                    };
                    for r in 1..nz {
                        let k = r;
                        dk[k] = inv_norm(dk[k], [di[k], dj[k], dk[k - 1]], [isign, jsign, 1.]);
                        let k = nz - 1 - r;
                        dk[k] = inv_norm(dk[k], [di[k], dj[k], dk[k + 1]], [isign, jsign, -1.]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min;

    fn check_connectivity_2d((ci, cj): (usize, usize), dim: (usize, usize), sign: [f64; 2]) {
        let correct_array = {
            let mut u_array = Array::zeros(dim);
            for ((i, j), u) in u_array.indexed_iter_mut() {
                *u = if ((sign[0] == 1. && i >= ci) || (sign[0] == -1. && i <= ci))
                    && ((sign[1] == 1. && j >= cj) || (sign[1] == -1. && j <= cj))
                {
                    0.
                } else {
                    1.
                }
            }
            u_array
        };

        let mut d_array = Array::from_elem(dim, 1.);
        d_array[(ci, cj)] = 0.;

        fast_sweep_2d(d_array.as_slice_mut().unwrap(), dim, |d, v, s| {
            if s == sign {
                min(d, min(v[0], v[1]))
            } else {
                d
            }
        });
        let tol = 1e-13;
        if !d_array.all_close(&correct_array, tol) {
            let mut diff =
                (d_array.clone() - correct_array).map(|v| if v.abs() >= tol { 1 } else { 0 });
            diff[(ci, cj)] = 4;
            panic!(
                "Arrays not close for direction {:?}!\nDiff:\n{:?}\nGot:\n{:.0?}\n",
                sign, diff, d_array
            );
        }
    }

    /// Check propagation of information.
    #[test]
    fn fast_sweep_2d_connectivity() {
        for &(ni, nj) in &[(4, 5), (5, 4), (5, 5)] {
            for ci in 0..ni {
                for cj in 0..nj {
                    for &s in &[[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]] {
                        check_connectivity_2d((ci, cj), (ni, nj), s);
                    }
                }
            }
        }
    }

    fn check_directionality_2d((ci, cj): (usize, usize), dim: (usize, usize), dir: u32) {
        let correct_array = {
            let mut u_array = Array::zeros(dim);
            for ((i, j), u) in u_array.indexed_iter_mut() {
                *u = match dir {
                    0 if i <= ci && j == cj => 0.,
                    1 if i >= ci && j == cj => 0.,
                    2 if i == ci && j <= cj => 0.,
                    3 if i == ci && j >= cj => 0.,
                    _ => 1.,
                }
            }
            u_array
        };

        let mut d_array = Array::from_elem(dim, 1.);
        d_array[(ci, cj)] = 0.;

        fast_sweep_2d(d_array.as_slice_mut().unwrap(), dim, |d, v, s| {
            min(
                d,
                match dir {
                    0 if s[0] == -1. => v[0],
                    1 if s[0] == 1. => v[0],
                    2 if s[1] == -1. => v[1],
                    3 if s[1] == 1. => v[1],
                    _ => std::f64::MAX,
                },
            )
        });
        let tol = 1e-13;
        if !d_array.all_close(&correct_array, tol) {
            let mut diff =
                (d_array.clone() - correct_array).map(|v| if v.abs() >= tol { 1 } else { 0 });
            diff[(ci, cj)] = 4;
            panic!(
                "Arrays not close for direction {:?}!\nDiff:\n{:?}\nGot:\n{:.0?}\n",
                dir, diff, d_array
            );
        }
    }

    /// Check that i and j components are passed correctly .
    #[test]
    fn fast_sweep_2d_directionality() {
        for &(ni, nj) in &[(4, 5), (5, 4), (5, 5)] {
            for ci in 0..ni {
                for cj in 0..nj {
                    for s in 0..4 {
                        check_directionality_2d((ci, cj), (ni, nj), s);
                    }
                }
            }
        }
    }
}
