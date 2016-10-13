//! The fast sweeping method for the computation of the signed distance function in 2D in 3D.
//!
//! Based on [1].
//!
//! ## References
//!
//! [1] Zhao, Hongkai A fast sweeping method for eikonal equations. Math. Comp. 74 (2005), no. 250,
//! 603–627.

mod level_set;
mod eikonal;

/// Implementation of min that compiles to the `minsd` instruction on intel.
#[inline(always)]
pub fn min(x: f64, y: f64) -> f64 {
    if x > y {
        y
    } else {
        x
    }
}

/// Computes the signed distance from the _zero_ level set of the function given by the values of
/// `u` on a regular 3D grid of dimensions `dim` and stores the result in a preallocated array `d`.
///
/// The zero level set is reconstructed by splitting each cube of the grid into 6 tetrahedra whose
/// vertices have coordinates `(0, 0, 0)`, ..., ..., `(1, 1, 1)` (relative to the cube), and the
/// following vertex in the sequence flips exactly one coordinate from `0` to `1`. The level set
/// function is then assumed to be linear on the tetrahedron.
///
/// `h` is the distance between neighboring nodes.
///
/// `u` is assumed to be in the _row-major_ order (C order).
///
/// Returns `std::f64::MAX` if all `u` are positive and `-std::f64::MAX` if all `u` are negative.
pub fn signed_distance_3d(d: &mut [f64], u: &[f64], dim: (usize, usize, usize), h: f64) {
    assert_eq!(dim.0 * dim.1 * dim.2, u.len());
    assert_eq!(dim.0 * dim.1 * dim.2, d.len());
    level_set::init_dist_3d(d, u, dim);
    eikonal::fast_sweep_dist_3d(d, dim);

    // compute the signed distance function from the solution of the eikonal equation
    for i in 0..d.len() {
        if u[i] < 0. {
            d[i] = -d[i] * h;
        } else {
            d[i] *= h;
        }
    }
}

/// Computes the signed distance from the _zero_ level set of the function given by the values of
/// `u` on a regular 2D grid of dimensions `dim` and stores the result in a preallocated array `d`.
///
/// The zero level set is reconstructed by splitting each square of the grid into 2 triangles whose
/// vertices have coordinates `(0, 0)`, ..., `(1, 1)` (relative to the square), and the
/// following vertex in the sequence flips exactly one coordinate from `0` to `1`. The level set
/// function is then assumed to be linear on the triangle.
///
/// ```text,ignore
///   0,1 *----* 1,1
///       |   /|
///       |  / |
///       | /  |
///       |/   |
///   0,0 *----* 1,0
/// ```
///
/// `h` is the distance between neighboring nodes.
///
/// `u` is assumed to be in the _row-major_ order (C order).
///
/// Returns `std::f64::MAX` if all `u` are nonnegative (`-std::f64::MAX` if all `u` are negative).
pub fn signed_distance_2d(d: &mut [f64], u: &[f64], dim: (usize, usize), h: f64) {
    assert_eq!(dim.0 * dim.1, u.len());
    assert_eq!(dim.0 * dim.1, d.len());
    level_set::init_dist_2d(d, u, dim);
    eikonal::fast_sweep_dist_2d(d, dim);

    // compute the signed distance function from the solution of the eikonal equation
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

    fn check_line(gx: f64, gy: f64, c: f64, dim: (usize, usize), tol: f64, print: bool) -> bool {
        let (nx, ny) = dim;
        let xs = Array::linspace(0., 1., nx);
        let ys = Array::linspace(0., (ny - 1) as f64 / (nx - 1) as f64, ny);
        let u_array = {
            let mut u_array = Array::zeros(dim);
            for ((i, j), u) in u_array.indexed_iter_mut() {
                let (x, y) = (xs[i], ys[j]);
                *u = x * gx + y * gy + c;
            }
            u_array
        };
        let u = u_array.as_slice().unwrap();

        let d = {
            let mut d = vec![0f64; nx * ny];
            signed_distance_2d(&mut d, &u, dim, 1. / (nx - 1) as f64);
            Array::from_shape_vec(dim, d).unwrap()
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
            check_line(0., 1., -((y - y.floor()) * 0.9 + 0.05), (9, 17), 0.00001, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_y_axis_line() {
        fn prop(x: f64) -> bool {
            check_line(1., 0., -((x - x.floor()) * 0.9 + 0.05), (16, 9), 0.00001, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_diagonal() {
        assert!(check_line((0.5f64).sqrt(),
                           (0.5f64).sqrt(),
                           -(0.5f64).sqrt(),
                           (9, 9),
                           1e-6,
                           false));
        assert!(check_line(-(0.5f64).sqrt(), (0.5f64).sqrt(), 0., (9, 9), 1e-6, false));
    }

    #[test]
    fn it_preserves_lines() {
        fn prop(ta: f64) -> bool {
            let n = 17;
            let ta = (ta - ta.floor()) * 2. * ::std::f64::consts::PI;
            let (gy, gx) = ta.sin_cos();
            let c = -(gx + gy) * 0.5;

            let xs = Array::linspace(0., 1., n);
            let ys = Array::linspace(0., 1., n);
            let u_array = {
                let mut u_array = xs.broadcast((n, n)).unwrap().to_owned();
                u_array.zip_mut_with(&ys.broadcast((n, n)).unwrap().t(),
                                     |x, y| *x = *x * gx + *y * gy + c);
                u_array
            };
            let u = u_array.as_slice().unwrap();

            let d = {
                let mut d = vec![0f64; n * n];
                signed_distance_2d(&mut d, &u, (n, n), 1. / (n - 1) as f64);
                Array::from_shape_vec((n, n), d).unwrap()
            };
            let d2 = {
                let mut d2 = vec![0f64; n * n];
                signed_distance_2d(&mut d2, d.as_slice().unwrap(), (n, n), 1. / (n - 1) as f64);
                Array::from_shape_vec((n, n), d2).unwrap()
            };
            // check only elements away from the boundary
            let s = &[Si(2, Some(-2), 1), Si(2, Some(-2), 1)];
            d.slice(s).all_close(&d2.slice(s), 0.001)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    fn check_plane(gx: f64, gy: f64, gz: f64, c: f64, dim: (usize, usize, usize), tol: f64, print: bool) -> bool {
        let (nx, ny, nz) = dim;
        let xs = Array::linspace(0., 1., nx);
        let ys = Array::linspace(0., (ny - 1) as f64 / (nx - 1) as f64, ny);
        let zs = Array::linspace(0., (nz - 1) as f64 / (nx - 1) as f64, nz);
        let u_array = {
            let mut u_array = Array::zeros(dim);
            for ((i, j, k), u) in u_array.indexed_iter_mut() {
                let (x, y, z) = (xs[i], ys[j], zs[k]);
                *u = x * gx + y * gy + z * gz + c;
            }
            u_array
        };
        let u = u_array.as_slice().unwrap();

        let d = {
            let mut d = vec![0f64; nx * ny * nz];
            signed_distance_3d(&mut d, &u, dim, 1. / (nx - 1) as f64);
            Array::from_shape_vec(dim, d).unwrap()
        };
        if print {
            println!("{}", u_array);
            println!("{}", d);
        }
        d.all_close(&u_array, tol)
    }

    #[test]
    fn it_works_for_x_axis_plane() {
        fn prop(x: f64) -> bool {
            check_plane(1., 0., 0., -((x - x.floor()) * 0.9 + 0.05), (9, 14, 18), 1e-6, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_y_axis_plane() {
        fn prop(x: f64) -> bool {
            check_plane(0., 1., 0., -((x - x.floor()) * 0.9 + 0.05), (9, 14, 18), 1e-6, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_z_axis_plane() {
        fn prop(x: f64) -> bool {
            check_plane(0., 0., 1., -((x - x.floor()) * 0.9 + 0.05), (9, 14, 18), 1e-6, false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }
}
