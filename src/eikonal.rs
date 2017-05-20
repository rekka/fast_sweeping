use super::min;

macro_rules! min_of {
    ( $d:ident, $s:expr, $i:expr, $ni:expr, $stride:expr ) => {
        if $i == 0 {
            $d[$s + $stride]
        } else if $i == $ni - 1 {
            $d[$s - $stride]
        } else {
            min($d[$s - $stride],$d[$s + $stride])
        };
    }
}

pub fn fast_sweep_anisotropic_dist_3d<F>(d: &mut [f64],
                                         dim: (usize, usize, usize),
                                         mut inv_dual_norm: F)
    where F: FnMut(f64, [f64; 3], [f64; 3]) -> f64
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

                        d[s] = inv_dual_norm(d[s],
                                             [d[ip * sx + j * sy + k * sz],
                                              d[i * sx + jp * sy + k * sz],
                                              d[i * sx + j * sy + kp * sz]],
                                             [isign, jsign, ksign]);
                    }
                } else {
                    let ksign = 1.;
                    for k in 1..nz {
                        let kp = k - 1;
                        let s = i * sx + j * sy + k * sz;

                        d[s] = inv_dual_norm(d[s],
                                             [d[ip * sx + j * sy + k * sz],
                                              d[i * sx + jp * sy + k * sz],
                                              d[i * sx + j * sy + kp * sz]],
                                             [isign, jsign, ksign]);
                    }
                }
            }
        }
    }

}
/// Computes the solution of the eikonal equation in 2D using the Fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
pub fn fast_sweep_dist_3d(d: &mut [f64], dim: (usize, usize, usize)) {
    let (nx, ny, nz) = dim;
    assert_eq!(nx * ny * nz, d.len());
    let (sx, sy, sz) = (ny * nz, nz, 1);
    // sweep in 8 directions
    for m in 0..8 {
        for p in 0..nx {
            let i = if m & 0b001 == 0 { nx - 1 - p } else { p };
            for q in 0..ny {
                let j = if m & 0b010 == 0 { ny - 1 - q } else { q };
                for r in 0..nz {
                    let k = if m & 0b100 == 0 { nz - 1 - r } else { r };

                    let s = i * sx + j * sy + k * sz;

                    // get and order a, b, c
                    let (a, b, c) = {
                        use std::mem::swap;
                        let mut a = min_of!(d, s, i, nx, sx);
                        let mut b = min_of!(d, s, j, ny, sy);
                        let mut c = min_of!(d, s, k, nz, sz);

                        if a > b {
                            swap(&mut a, &mut b);
                        }
                        if b > c {
                            swap(&mut b, &mut c);
                        }
                        if a > b {
                            swap(&mut a, &mut b);
                        }

                        (a, b, c)
                    };

                    let x = if b >= a + 1. {
                        a + 1.
                    } else {
                        let x = 0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt());
                        if x <= c {
                            x
                        } else {
                            let v = (1. / 3.) *
                                    (a + b + c +
                                     (3. + (a + b + c).powi(2) - 3. * (a * a + b * b + c * c))
                                         .sqrt());
                            v
                        }
                    };

                    d[s] = min(d[s], x);
                }
            }
        }
    }
}

/// Computes the solution of the eikonal equation in 2D using the Fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
pub fn fast_sweep_dist_2d(d: &mut [f64], dim: (usize, usize)) {
    let (nx, ny) = dim;
    let (sx, sy) = (ny, 1);
    assert_eq!(nx * ny, d.len());
    // sweep in 4 directions
    for m in 0..4 {
        for p in 0..nx {
            let i = if m & 0b001 == 0 { nx - 1 - p } else { p };
            for q in 0..ny {
                let j = if m & 0b010 == 0 { ny - 1 - q } else { q };

                let s = i * sx + j * sy;
                let a = min_of!(d, s, i, nx, sx);
                let b = min_of!(d, s, j, ny, sy);

                let x = if (a - b).abs() >= 1. {
                    min(a, b) + 1.
                } else {
                    0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt())
                };

                d[s] = min(d[s], x);
            }
        }
    }
}

/// Fast sweeping method for a general anisotropic norm.
///
/// `inv_dual_norm(d, [d1, d2], [s1, s2]) -> t` needs to solve the "inverse problem" for the norm:
/// Given values `d_i` at points `-s_1 e_1`, find the largest value `t ≤ d` at the origin such that
/// `||p|| ≤ 1`, where `p_i = (s_i (t - d_i))_+` and `||p||` is the __dual__ anisotropic norm.
pub fn fast_sweep_anisotropic_dist_2d<F>(d: &mut [f64], dim: (usize, usize), mut inv_dual_norm: F)
    where F: FnMut(f64, [f64; 2], [f64; 2]) -> f64
{
    let (nx, ny) = dim;
    let (sx, sy) = (ny, 1);
    assert_eq!(nx * ny, d.len());
    // sweep in 4 directions
    for m in 0..4 {
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

                let s = i * sx + j * sy;

                d[s] = inv_dual_norm(d[s],
                                     [d[ip * sx + j * sy], d[i * sx + jp * sy]],
                                     [isign, jsign]);
            }
        }
    }
}
