//! The fast sweeping method for the eikonal equation.
//!
//! This is an implementation of the fast sweeping method proposed by Zhao (2004). It solves the
//! eikonal equation
//!
//! ‖∇u‖ = 1,
//!
//! where ‖.‖ denotes a norm, by performing 2^N Gauss-Seidel sweeps in alternating directions. Here
//! N is the dimension. Zhao showed that the accuracy for the Euclidean norm is O(h log h).

/// Computes the solution of the eikonal equation in 3D using the fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
///
/// `inv_dual_norm(d, [d1, d2, d3], [s1, s2, s3]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values `d_i` at points `-s_i e_i`, find the largest value `t ≤ d` at the origin such that
/// `||p|| ≤ 1`, where `p_i = (s_i (t - d_i))_+` and `||p||` is the __dual__ anisotropic norm.
pub fn fast_sweep_3d<F>(d: &mut [f64], dim: (usize, usize, usize), mut inv_dual_norm: F)
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
                        dk[k] = inv_dual_norm(dk[k], [di[k], dj[k], dk[k - 1]], [isign, jsign, 1.]);
                        let k = nz - 1 - r;
                        dk[k] =
                            inv_dual_norm(dk[k], [di[k], dj[k], dk[k + 1]], [isign, jsign, -1.]);
                    }
                }
            }
        }
    }
}

/// Computes the solution of the eikonal equation in 2D using the fast sweeping algorithm.
///
/// `inv_dual_norm(d, [d1, d2], [s1, s2]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values `d_i` at points `-s_i e_i`, find the largest value `t ≤ d` at the origin such that
/// `||p|| ≤ 1`, where `p_i = (s_i (t - d_i))_+` and `||p||` is the __dual__ anisotropic norm.
pub fn fast_sweep_2d<F>(d: &mut [f64], dim: (usize, usize), mut inv_dual_norm: F)
where
    F: FnMut(f64, [f64; 2], [f64; 2]) -> f64,
{
    let (nx, ny) = dim;
    let (sx, _sy) = (ny, 1);
    assert_eq!(nx * ny, d.len());
    // sweep in 4 directions
    // for m in 0..4 {
    //     for p in 1..nx {
    //         let (i, ip, isign) = if m & 0b001 == 0 {
    //             (nx - 1 - p, nx - 1 - p + 1, -1.)
    //         } else {
    //             (p, p - 1, 1.)
    //         };
    //         for q in 1..ny {
    //             let (j, jp, jsign) = if m & 0b010 == 0 {
    //                 (ny - 1 - q, ny - 1 - q + 1, -1.)
    //             } else {
    //                 (q, q - 1, 1.)
    //             };
    //
    //             let s = i * sx + j * sy;
    //
    //             d[s] = inv_dual_norm(
    //                 d[s],
    //                 [d[ip * sx + j * sy], d[i * sx + jp * sy]],
    //                 [isign, jsign],
    //             );
    //         }
    //     }
    // }

    // sweep in 4 directions
    // try to avoid bounds checks and impove cache locality (about 45% faster than the above version)
    for p in 1..nx {
        let (di, dj) = d.split_at_mut(p * sx);
        let di = &di[(p - 1) * sx..][..ny];
        let dj = &mut dj[..ny];
        for q in 1..ny {
            let j = q;
            // Two independent computations (both forward and reversed sweep at once). Instruction
            // level parallelism! Major performance gain.
            dj[j] = inv_dual_norm(dj[j], [di[j], dj[j - 1]], [1., 1.]);
            // rust 1.30 cannot optimize away 2 bounds checks here :(
            let j = ny - 1 - q;
            dj[j] = inv_dual_norm(dj[j], [di[j], dj[j + 1]], [1., -1.]);
        }
        // The following code avoids bounds checks, but cannot exploit instruction level parallelism in the
        // above loop.
        // let mut prev = dj[0];
        // for (dj, di) in (&mut dj[1..]).into_iter().zip(di[1..].into_iter()) {
        //     prev = inv_dual_norm(*dj, [*di, prev], [1., 1.]);
        //     *dj = prev;
        // }
        // let mut prev = dj[ny - 1];
        // for (dj, di) in (&mut dj[..ny-1]).into_iter().zip(di[..ny-1].into_iter()).rev() {
        //     prev = inv_dual_norm(*dj, [*di, prev], [1., -1.]);
        //     *dj = prev;
        // }
    }
    for p in (0..nx - 1).rev() {
        let (dj, di) = d.split_at_mut((p + 1) * sx);
        let di = &di[..ny];
        let dj = &mut dj[p * sx..][..ny];
        for q in 1..ny {
            let j = q;
            dj[j] = inv_dual_norm(dj[j], [di[j], dj[j - 1]], [-1., 1.]);
            let j = ny - 1 - q;
            dj[j] = inv_dual_norm(dj[j], [di[j], dj[j + 1]], [-1., -1.]);
        }
    }
}
