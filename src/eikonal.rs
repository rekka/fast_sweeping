macro_rules! min_of {
    ( $d:ident, $s:expr, $i:expr, $ni:expr, $stride:expr ) => {
        if $i == 0 {
            $d[$s + $stride]
        } else if $i == $ni - 1 {
            $d[$s - $stride]
        } else {
            $d[$s - $stride].min($d[$s + $stride])
        };
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
            let i = if m & 0b001 == 0 {
                nx - 1 - p
            } else {
                p
            };
            for q in 0..ny {
                let j = if m & 0b010 == 0 {
                    ny - 1 - q
                } else {
                    q
                };
                for r in 0..nz {
                    let k = if m & 0b100 == 0 {
                        nz - 1 - r
                    } else {
                        r
                    };

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
                             (3. + (a + b + c).powi(2) - 3. * (a * a + b * b + c * c)).sqrt());
                            v
                        }
                    };

                    d[s] = d[s].min(x);
                }
            }
        }
    }
}

/// Computes the solution of the eikonal equation in 2D using the Fast sweeping algorithm.
///
/// `d` should be initialized to a large value at the unknown nodes.
pub fn fast_sweep_dist(d: &mut [f64], dim: (usize, usize)) {
    let (nx, ny) = dim;
    assert_eq!(nx * ny, d.len());
    // sweep in 4 directions
    for k in 1..5 {
        for q in 0..ny {
            let j = match k {
                3 | 4 => ny - 1 - q,
                _ => q,
            };
            for p in 0..nx {
                let i = match k {
                    2 | 3 => nx - 1 - p,
                    _ => p,
                };
                let s = j * nx + i;
                let a = if i == 0 {
                    d[s + 1]
                } else if i == nx - 1 {
                    d[s - 1]
                } else {
                    d[s - 1].min(d[s + 1])
                };
                let b = if j == 0 {
                    d[s + nx]
                } else if j == ny - 1 {
                    d[s - nx]
                } else {
                    d[s - nx].min(d[s + nx])
                };
                let x = if (a - b).abs() >= 1. {
                    a.min(b) + 1.
                } else {
                    0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt())
                };

                d[s] = d[s].min(x);
            }
        }
    }
}
