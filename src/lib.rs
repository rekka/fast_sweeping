//! The fast sweeping method for the computation of the signed distance function in 2D.
//!
//!

/// Computes the signed distance function from a line segment given as the _zero_ level set of a
/// linear function on an isosceles right-angle triangle.
///
/// Inputs are `u`, the values at the verteces. The vertex 0 is the one with the right angle.
///
/// The function returns the values of the signed distance function or `None` if the zero level set
/// does not pass through the triangle.
pub fn triangle_dist(u: [f64; 3]) -> Option<[f64; 3]> {
    let mut u = u;
    // normalize so that u[0] >= 0.
    if u[0] < 0. {
        for u in u.iter_mut() {
            *u = -*u;
        }
    }

    // gradient vector
    let gx = u[1] - u[0];
    let gy = u[2] - u[0];
    let g_norm = (gx * gx + gy * gy).sqrt();

    if u[1] >= 0. {
        if u[2] >= 0. {
            // well isn't this ugly
            return match (u[0], u[1], u[2]) {
                (0., 0., 0.) => Some([0., 0., 0.]),
                (_, 0., 0.) => Some([(0.5f64).sqrt(), 0., 0.]),
                (0., _, 0.) => Some([0., 1., 0.]),
                (0., 0., _) => Some([0., 0., 1.]),
                (0., _, _) => Some([0., 1., 1.]),
                (_, 0., _) => Some([1., 0., (2f64).sqrt()]),
                (_, _, 0.) => Some([1., (2f64).sqrt(), 0.]),
                    _ => None
            };
        } else {
            // u[2] < 0.
            // intersect position
            let i02 = u[0] / (u[0] - u[2]);
            let i12 = (2f64).sqrt() * u[1] / (u[1] - u[2]);
            // find the direction of the gradient
            // to deduce the vertex that is closest to the line
            if gx <= 0. {
                // 0
                return Some([u[0] / g_norm, i12, 1. - i02]);
            } else if gx > -gy {
                // 1
                return Some([i02, u[1] / g_norm, (2f64).sqrt() - i12]);
            } else {
                // 2
                return Some([i02, i12, -u[2] / g_norm]);
            }
        }
    } else {
        // u[1] < 0.
        if u[2] >= 0. {
            // intersect position
            let i01 = u[0] / (u[0] - u[1]);
            let i12 = (2f64).sqrt() * u[1] / (u[1] - u[2]);
            // find the direction of the gradient
            // to deduce the vertex that is closest to the line
            if gy <= 0. {
                // 0
                return Some([u[0] / g_norm, 1. - i01, (2f64).sqrt() - i12]);
            } else if -gx > gy {
                // 1
                return Some([i01, -u[1] / g_norm, (2f64).sqrt() - i12]);
            } else {
                // 2
                return Some([i01, i12, u[2] / g_norm]);
            }
        } else {
            // u[2] < 0.
            // intersect position
            let i10 = u[1] / (u[1] - u[0]);
            let i20 = u[2] / (u[2] - u[0]);

            return Some([u[0] / g_norm, i10, i20]);
        }
    }
}

/// Initializes distance around the free boundary.
///
/// Splits every square into two triangles and computes the distance on each of them.
pub fn init_dist(d: &mut [f64], u: &[f64], dim: (usize, usize)) {
    let (nx, ny) = dim;
    assert_eq!(nx * ny, u.len());
    assert_eq!(nx * ny, d.len());

    for i in 0..d.len() {
        d[i] = std::f64::MAX;
    }

    for j in 1..ny {
        for i in 1..nx {
            let s = j * nx + i;
            let r = triangle_dist([u[s - nx - 1], u[s - nx], u[s - 1]]);
            if let Some(e) = r {
                d[s - nx - 1] = e[0].min(d[s - nx - 1]);
                d[s - nx] = e[1].min(d[s - nx]);
                d[s - 1] = e[2].min(d[s - 1]);
            }
            let r = triangle_dist([u[s], u[s - nx], u[s - 1]]);
            if let Some(e) = r {
                d[s] = e[0].min(d[s]);
                d[s - nx] = e[1].min(d[s - nx]);
                d[s - 1] = e[2].min(d[s - 1]);
            }
        }
    }
}

/// Computes the solution of the eikonal equation in 2D using the Fast Sweeping algorithm.
///
/// `d` should be initialized to large values at the unknown nodes.
pub fn fast_sweep_dist(d: &mut [f64], dim: (usize, usize)) {
    let (nx, ny) = dim;
    assert_eq!(nx * ny, d.len());
    // sweep in 4 directions
    for k in 1..5 {
        for q in 0..ny {
            let j = match k {
                3 | 4 => ny - 1 - q,
                _ => q
            };
            for p in 0..nx {
                let i = match k {
                    2 | 3 => nx - 1 - p,
                    _ => p
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

/// Computes the signed distance from the _zero_ level set of the function given by the values of
/// `u` on a regular grid of dimensions `dim` and stores the result to a preallocated array `d`.
///
/// `h` is the distance between neighboring nodes.
///
/// Returns `std::f64::MAX` if all `u` are positive (`-std::f64::MAX` if all `u` are negative).
pub fn signed_distance(d: &mut [f64], u: &[f64], dim: (usize, usize), h: f64) {
    assert_eq!(dim.0 * dim.1, u.len());
    assert_eq!(dim.0 * dim.1, d.len());
    init_dist(d, u, dim);
    fast_sweep_dist(d, dim);

    for i in 0..d.len() {
        if u[i] < 0. {
            d[i] = -d[i] * h;
        } else {
            d[i] *= h;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    extern crate quickcheck;
    extern crate ndarray;
    use self::quickcheck::quickcheck;
    use self::ndarray::prelude::*;
    use self::ndarray::Si;

    fn check_line(gx: f64, gy: f64, c: f64, n: usize, tol: f64, print: bool) -> bool {
            let xs = OwnedArray::linspace(0., 1., n);
            let ys = OwnedArray::linspace(0., 1., n);
            let u_array = {
                let mut u_array = xs.broadcast((n, n)).unwrap().to_owned();
                u_array.zip_mut_with(&ys.broadcast((n, n)).unwrap().t(),
                                     |x, y| *x = *x * gx + *y * gy + c);
                u_array
            };
            let u = u_array.as_slice().unwrap();

            let d = {
                let mut d = vec![0f64; n * n];
                signed_distance(&mut d, &u, (n, n), 1. / (n - 1) as f64);
                OwnedArray::from_shape_vec((n, n), d).unwrap()
            };
            if print {
                println!("{}", u_array);
                println!("{}", d);
            }
            d.all_close(&u_array, tol)
    }

    #[test]
    fn it_works_for_x_axis_line() {
        fn prop(y: f64) -> bool {
            check_line(0., 1., -((y - y.floor()) * 0.9 + 0.05), 9, 0.00001, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_y_axis_line() {
        fn prop(x: f64) -> bool {
            check_line(1., 0., -((x - x.floor()) * 0.9 + 0.05), 9, 0.00001, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_diagonal() {
        assert!(check_line((0.5f64).sqrt(),(0.5f64).sqrt(), -(0.5f64).sqrt(), 9, 1e-6, false));
        assert!(check_line(-(0.5f64).sqrt(),(0.5f64).sqrt(), 0., 9, 1e-6, false));
    }

    #[test]
    fn it_preserves_lines() {
        fn prop(ta: f64) -> bool {
            let n = 17;
            let ta = (ta - ta.floor()) * 2. * ::std::f64::consts::PI;
            let (gy, gx) = ta.sin_cos();
            let c = -(gx + gy) * 0.5;

            let xs = OwnedArray::linspace(0., 1., n);
            let ys = OwnedArray::linspace(0., 1., n);
            let u_array = {
                let mut u_array = xs.broadcast((n, n)).unwrap().to_owned();
                u_array.zip_mut_with(&ys.broadcast((n, n)).unwrap().t(),
                                     |x, y| *x = *x * gx + *y * gy + c);
                u_array
            };
            let u = u_array.as_slice().unwrap();

            let d = {
                let mut d = vec![0f64; n * n];
                signed_distance(&mut d, &u, (n, n), 1. / (n - 1) as f64);
                OwnedArray::from_shape_vec((n, n), d).unwrap()
            };
            let d2 = {
                let mut d2 = vec![0f64; n * n];
                signed_distance(&mut d2, d.as_slice().unwrap(), (n, n), 1. / (n - 1) as f64);
                OwnedArray::from_shape_vec((n, n), d2).unwrap()
            };
            // check only elements away from the boundary
            let s = &[Si(2, Some(-2), 1), Si(2, Some(-2), 1)];
            d.slice(s).all_close(&d2.slice(s), 0.001)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_is_nonnegative_on_triangle() {
        fn prop(x: f64, y: f64, z: f64) -> bool {
            if let Some(e) = triangle_dist([x, y, z]) {
                e.into_iter().all(|&x| x >= 0.)
            } else {
                true
            }
        }
        quickcheck(prop as fn(f64, f64, f64) -> bool);
    }

    #[test]
    fn simple_triangles() {
        assert_eq!(triangle_dist([0., 0., 0.]), Some([0., 0., 0.]));
        assert_eq!(triangle_dist([-1., 0., 0.]), Some([(0.5f64).sqrt(), 0., 0.]));
        assert_eq!(triangle_dist([0., 1., 0.]), Some([0., 1., 0.]));
        assert_eq!(triangle_dist([0., -1., -1.]), Some([0., 1., 1.]));
        assert_eq!(triangle_dist([0., 1., 1.]), Some([0., 1., 1.]));
        assert_eq!(triangle_dist([1., 1., 0.]), Some([1., (2f64).sqrt(), 0.]));
    }
}
