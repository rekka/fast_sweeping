//! Implementation of the fast sweeping method.

/// Computes the solution of the eikonal equation in 3D using the fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
///
/// `inv_dual_norm(d, [d1, d2, d3], [s1, s2, s3]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values `d_i` at points `-s_i e_i`, find the largest value `t ≤ d` at the origin such that
/// `||p|| ≤ 1`, where `p_i = (s_i (t - d_i))_+` and `||p||` is the __dual__ anisotropic norm.
pub fn fast_sweep_3d<F>(
    d: &mut [f64],
    dim: (usize, usize, usize),
    mut inv_dual_norm: F,
) where
    F: FnMut(f64, [f64; 3], [f64; 3]) -> f64,
{
    let (nx, ny, nz) = dim;
    assert_eq!(nx * ny * nz, d.len());
    let (sx, sy, sz) = (ny * nz, nz, 1);
    // sweep in 8 directions
    for m in 0..8 {
        for p in 1..nx {
            let (i, ip, isign) = if m & 0b001 == 0 {
                (nx - 1 - p, nx - 1 - p + 1, -1.)
            } else {
                (p, p - 1, 1.)
            };
            for q in 1..ny {
                let (j, jp, jsign) = if m & 0b010 == 0 {
                    (ny - 1 - q, ny - 1 - q + 1, -1.)
                } else {
                    (q, q - 1, 1.)
                };

                if m & 0b100 == 0 {
                    let ksign = -1.;
                    for k in (0..nz - 1).rev() {
                        let kp = k + 1;
                        let s = i * sx + j * sy + k * sz;

                        d[s] = inv_dual_norm(
                            d[s],
                            [
                                d[ip * sx + j * sy + k * sz],
                                d[i * sx + jp * sy + k * sz],
                                d[i * sx + j * sy + kp * sz],
                            ],
                            [isign, jsign, ksign],
                        );
                    }
                } else {
                    let ksign = 1.;
                    for k in 1..nz {
                        let kp = k - 1;
                        let s = i * sx + j * sy + k * sz;

                        d[s] = inv_dual_norm(
                            d[s],
                            [
                                d[ip * sx + j * sy + k * sz],
                                d[i * sx + jp * sy + k * sz],
                                d[i * sx + j * sy + kp * sz],
                            ],
                            [isign, jsign, ksign],
                        );
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
