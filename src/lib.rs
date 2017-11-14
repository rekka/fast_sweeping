//! The fast sweeping method for the computation of the signed distance function in 2D in 3D.
//!
//! Based on [1].
//!
//! ## Usage
//!
//! Add the following to your `Cargo.toml`
//!
//! ```toml
//! [dependencies.fast_sweeping]
//! git = "https://github.com/rekka/fast_sweeping.git"
//! ```
//!
//! At the top of your crate add:
//!
//! ```rust
//! extern crate fast_sweeping;
//! ```
//!
//! Depending on the dimension, use `signed_distance_2d` or `signed_distance_3d`.
//!
//! ## Accuracy
//!
//! There are two main things to consider when evaluating the accuracy of the method.
//!
//! ### Initialization near the level set
//!
//! There is no unique way of doing this. In this implementation, we simply split the squares/cubes
//! of the regular mesh into 2 triangles/6 tetrahedra and assume that the level set function is
//! linear on each of them. The initial distance at the vertices is then taken as the distance to
//! the line/plane going through the triangle/tetrahedron (_not_ to the intersection of the
//! line/plane with the triangle/tetrahedron). If the vertex is contained in multiple
//! triangles/tetrahedra that intersect the level set, the minimum of all distances is taken.
//!
//! The main advantages of this approach are:
//!
//! - Simple.
//! - Does _not_ move _flat_ parts of the level set (unless near other parts of the level set).
//! - Does not move symmetric corners.
//!
//! Disadvantages:
//!
//! - Introduces some more anisotropy.
//! - The initial value outside of corners is not really the distance to the level set but smaller.
//! But this does not seem to actually cause larger error in the circle test case, see
//! `examples/error`. The error seems to be bigger _inside_ a circle. It appears that within small
//! neighborhood (dist <= 3h) of the level set the max error of order h².
//!
//! For an example see `examples/redistance`.
//!
//! ### Finite difference approximation
//!
//! Here we use the simplest first order upwind numerical discretization as given in [1]. This
//! gives an error of order `O(|h log h|)`.
//!
//! ## Performance
//!
//! The performance is limited by the speed of computing `sqrt` during the sweeps. Possible
//! future optimizations are:
//!
//!   - Only compute distance in a small neighborhood of the level set.
//!   - Compute two square roots in one instruction `sqrtpd`.
//!   - Use multiple threads. However, this is relatively nontrivial due to the sequential nature of
//!     the Gauss-Seidel iteration.
//!
//! ## References
//!
//! [1] Zhao, Hongkai A fast sweeping method for eikonal equations. Math. Comp. 74 (2005), no. 250,
//! 603–627.

extern crate isosurface;

pub mod level_set;
pub mod eikonal;
pub mod dist;

/// Implementation of min that compiles to the `minsd` instruction on intel.
#[inline(always)]
fn min(x: f64, y: f64) -> f64 {
    if x > y { y } else { x }
}

/// Implementation of max that compiles to the `maxsd` instruction on intel.
#[inline(always)]
fn max(x: f64, y: f64) -> f64 {
    if x < y { y } else { x }
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
    anisotropic_signed_distance_3d(d,
                                   u,
                                   dim,
                                   h,
                                   |p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(),
                                   |d, v, _| {
        let (a, b, c) = {
            use std::mem::swap;
            let mut a = v[0];
            let mut b = v[1];
            let mut c = v[2];

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

        min(d, x)
    });
}

pub fn max_signed_distance_3d(d: &mut [f64], u: &[f64], dim: (usize, usize, usize), h: f64) {
    anisotropic_signed_distance_3d(d,
                                   u,
                                   dim,
                                   h,
                                   |p| p[0].abs() + p[1].abs() + p[2].abs(),
                                   |d, v, _| {
        let (a, b, c) = {
            use std::mem::swap;
            let mut a = v[0];
            let mut b = v[1];
            let mut c = v[2];

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
        min(min(d, a + 1.),
            min(0.5 * (a + b + 1.), (1. / 3.) * (a + b + c + 1.)))
    });
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
    anisotropic_signed_distance_2d(d,
                                   u,
                                   dim,
                                   h,
                                   |p| (p[0] * p[0] + p[1] * p[1]).sqrt(),
                                   |d, v, _| {
        let a = v[0];
        let b = v[1];

        let x = if (a - b).abs() >= 1. {
            min(a, b) + 1.
        } else {
            0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt())
        };

        min(d, x)
    });
}

/// Signed distance in the max norm.
pub fn max_signed_distance_2d(d: &mut [f64], u: &[f64], dim: (usize, usize), h: f64) {
    anisotropic_signed_distance_2d(d, u, dim, h, |p| p[0].abs() + p[1].abs(), |d, v, _| {
        min(min(d, v[0] + 1.), min(v[1] + 1., 0.5 * (v[0] + v[1] + 1.)))
    });
}

/// Signed distance in the l¹ norm.
pub fn l1_signed_distance_2d(d: &mut [f64], u: &[f64], dim: (usize, usize), h: f64) {
    anisotropic_signed_distance_2d(d,
                                   u,
                                   dim,
                                   h,
                                   |p| max(p[0].abs(), p[1].abs()),
                                   |d, v, _| min(d, min(v[0], v[1]) + 1.));
}


/// Computes the anisotropic signed distance function.
///
/// `dual_norm` is the __dual__ norm. It must be a positively one-homogeneous function.
///
/// `inv_dual_norm(d, [d_1, d_2], [s_1, s_2]) -> t` needs to solve the "inverse problem" for the
/// norm: Given values `d_i` at points `-s_i e_i`, find the largest value `t ≤ d` at the origin
/// such that `||p|| ≤ 1`, where `p_i = (s_i (t - d_i))_+` and `||p||` is the __dual__ anisotropic
/// norm.
pub fn anisotropic_signed_distance_2d<N, INV>(d: &mut [f64],
                                              u: &[f64],
                                              dim: (usize, usize),
                                              h: f64,
                                              dual_norm: N,
                                              inv_dual_norm: INV)
    where N: FnMut([f64; 2]) -> f64,
          INV: FnMut(f64, [f64; 2], [f64; 2]) -> f64
{
    assert_eq!(dim.0 * dim.1, u.len());
    assert_eq!(dim.0 * dim.1, d.len());

    level_set::init_anisotropic_dist_2d(d, u, dim, dual_norm);
    eikonal::fast_sweep_anisotropic_dist_2d(d, dim, inv_dual_norm);

    // compute the signed distance function from the solution of the eikonal equation
    for i in 0..d.len() {
        if u[i] < 0. {
            d[i] = -d[i] * h;
        } else {
            d[i] *= h;
        }
    }
}

pub fn anisotropic_signed_distance_3d<N, INV>(d: &mut [f64],
                                              u: &[f64],
                                              dim: (usize, usize, usize),
                                              h: f64,
                                              dual_norm: N,
                                              inv_dual_norm: INV)
    where N: FnMut([f64; 3]) -> f64,
          INV: FnMut(f64, [f64; 3], [f64; 3]) -> f64
{
    assert_eq!(dim.0 * dim.1 * dim.2, u.len());
    assert_eq!(dim.0 * dim.1 * dim.2, d.len());

    level_set::init_anisotropic_dist_3d(d, u, dim, dual_norm);
    eikonal::fast_sweep_anisotropic_dist_3d(d, dim, inv_dual_norm);

    // compute the signed distance function from the solution of the eikonal equation
    for i in 0..d.len() {
        if u[i] < 0. {
            d[i] = -d[i] * h;
        } else {
            d[i] *= h;
        }
    }
}

pub mod legacy {
    use super::*;

    /// Original implementation (as in [Zhao])
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
            check_line(0.,
                       1.,
                       -((y - y.floor()) * 0.9 + 0.05),
                       (9, 17),
                       0.00001,
                       false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_y_axis_line() {
        fn prop(x: f64) -> bool {
            check_line(1.,
                       0.,
                       -((x - x.floor()) * 0.9 + 0.05),
                       (16, 9),
                       0.00001,
                       false)
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

    fn check_plane(gx: f64,
                   gy: f64,
                   gz: f64,
                   c: f64,
                   dim: (usize, usize, usize),
                   tol: f64,
                   print: bool)
                   -> bool {
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
            check_plane(1.,
                        0.,
                        0.,
                        -((x - x.floor()) * 0.9 + 0.05),
                        (9, 14, 18),
                        1e-6,
                        false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_y_axis_plane() {
        fn prop(x: f64) -> bool {
            check_plane(0.,
                        1.,
                        0.,
                        -((x - x.floor()) * 0.9 + 0.05),
                        (9, 14, 18),
                        1e-6,
                        false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }

    #[test]
    fn it_works_for_z_axis_plane() {
        fn prop(x: f64) -> bool {
            check_plane(0.,
                        0.,
                        1.,
                        -((x - x.floor()) * 0.9 + 0.05),
                        (9, 14, 18),
                        1e-6,
                        false)
        }
        quickcheck(prop as fn(f64) -> bool);
    }
}
