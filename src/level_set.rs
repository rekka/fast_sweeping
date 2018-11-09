//! Initialization of the signed distance function near the level set.
use super::min;
use std;

/// Computes the signed distance function from a plane given as the _zero_ level set of a linear
/// function on a tetrahedron with 4 vertices with unit coordinates starting at (0, 0, 0) and
/// ending at (1, 1, 1), and in between exactly one coordinate changes from 0 to 1.
///
/// Inputs are `u`, the values at the vertices.
///
/// `perm[a]` specifies on which step the coordinate `a` changes. For example, the first coordinate
/// changes from vertex `perm[0]` to vertex `perm[0] + 1`. Equivalently, `perm[0]` specifies which
/// edge of the tetrahedron is parallel to the first vector in the canonical basis, e₁. See
/// [`triangle_dist`](fn.triangle_dist.html) for more.
///
/// The function returns the values of the (non-signed) distance function or `None` if the zero
/// level set does not pass through the tetrahedron.
#[inline(always)]
fn tetrahedron_dist<F>(mut u: [f64; 4], mut dual_norm: F, perm: [usize; 3]) -> Option<[f64; 4]>
where
    F: FnMut([f64; 3]) -> f64,
{
    // iterator is a bit slower
    if (u[0] > 0. && u[1] > 0. && u[2] > 0. && u[3] > 0.)
        || (u[0] < 0. && u[1] < 0. && u[2] < 0. && u[3] < 0.)
    {
        return None;
    }

    let g = [u[1] - u[0], u[2] - u[1], u[3] - u[2]];
    let g = [g[perm[0]], g[perm[1]], g[perm[2]]];
    let norm = dual_norm(g);
    // everything is zero
    if norm == 0. {
        return Some([0.; 4]);
    }
    let g_norm_rcp = 1. / norm;

    // FIXME: this requires for norm to be even
    // Support triangular anisotropies?
    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// Initializes the distance function near the free boundary.
///
/// Splits every cube into six tetrahedra. Based on the level set function with values `u` given
/// on a regular grid, it computes the distance from the _zero_ level set in the nodes of the
/// tetrahedra through which the level set passes.  Stores the minimal value of the distance in the
/// preallocated slice `d`.
///
/// Nodes away from the boundary have their value set to `std::f64::MAX`.
///
/// `dual_norm` is the __dual__ norm. It must be an __even__ positively one-homogeneous function,
/// zero only at the origin.
pub fn init_dist_3d<F>(d: &mut [f64], u: &[f64], dim: (usize, usize, usize), mut dual_norm: F)
where
    F: FnMut([f64; 3]) -> f64,
{
    let (ni, nj, nk) = dim;
    assert_eq!(ni * nj * nk, u.len());
    assert_eq!(ni * nj * nk, d.len());
    let (si, sj, sk) = (nj * nk, nk, 1);

    for d in &mut *d {
        *d = std::f64::MAX;
    }

    macro_rules! tetra {
        ($s:expr, [$pi:expr, $pj:expr, $pk:expr]) => {
            // $pi specifies at which step the i-th coordinate changes, etc.
            let offset = |step|
                                        if $pi == step { si } else { 0 } +
                                        if $pj == step { sj } else { 0 } +
                                        if $pk == step { sk } else { 0 };

            let s0 = $s;
            let s1 = s0 - offset(0);
            let s2 = s1 - offset(1);
            let s3 = s2 - offset(2);
            let v = [u[s0], u[s1], u[s2], u[s3]];

            let r = tetrahedron_dist(v, &mut dual_norm, [$pi, $pj, $pk]);

            if let Some(r) = r {
                d[s0] = min(d[s0], r[0]);
                d[s1] = min(d[s1], r[1]);
                d[s2] = min(d[s2], r[2]);
                d[s3] = min(d[s3], r[3]);
            }
        };
    }

    for i in 1..ni {
        for j in 1..nj {
            let s = i * si + j * sj;
            let v = [u[s], u[s - si], u[s - sj], u[s - si - sj]];
            let mut all_pos_prev = v.iter().all(|&v| v > 0.);
            let mut all_neg_prev = v.iter().all(|&v| v < 0.);
            for k in 1..nk {
                let s = i * si + j * sj + k;
                let v = [u[s], u[s - si], u[s - sj], u[s - si - sj]];
                let mut all_pos = v.iter().all(|&v| v > 0.);
                let mut all_neg = v[0] < 0. && v[1] < 0. && v[2] < 0. && v[3] < 0.;

                if !((all_pos_prev && all_pos) || (all_neg_prev && all_neg)) {
                    tetra!(s, [0, 1, 2]);
                    tetra!(s, [0, 2, 1]);
                    tetra!(s, [1, 0, 2]);
                    tetra!(s, [2, 0, 1]);
                    tetra!(s, [1, 2, 0]);
                    tetra!(s, [2, 1, 0]);
                }
                all_pos_prev = all_pos;
                all_neg_prev = all_neg;
            }
        }
    }
}

/// Compute the anisotropic distance to the zero level set of a function on a axes-aligned
/// right triangle. The right angle is assumed to be at vertex 1.
///
/// `perm` specifies which leg of the triangle corresponds to the directions e₁ and e₂:
///
/// * `[0, 1]`: 0-1 leg is parallel to e₁, 1-2 leg is parallel to e₂
/// * `[1, 0]`: 0-1 leg is parallel to e₂, 1-2 leg is parallel to e₁
///
/// ```text
/// [0, 1]    [1, 0]
/// ------    ------
///
///     2      1--2         ^ e₂
///    /|      | /          |
///   / |      |/           |
///  0--1      0            +---> e₁
/// ```
///
fn triangle_dist<F>(mut u: [f64; 3], perm: [usize; 2], mut dual_norm: F) -> Option<[f64; 3]>
where
    F: FnMut([f64; 2]) -> f64,
{
    // check if sign differs (level set goes throught the triangle)
    if (u[0] > 0. && u[1] > 0. && u[2] > 0.) || (u[0] < 0. && u[1] < 0. && u[2] < 0.) {
        return None;
    }

    let g = [u[1] - u[0], u[2] - u[1]];
    let g = [g[perm[0]], g[perm[1]]];
    let norm = dual_norm(g);
    // all values are zero
    if norm == 0. {
        return Some([0., 0., 0.]);
    }
    let g_norm_rcp = 1. / norm;

    // TODO: This supports only even norms. Do we want to support triangle norms, for instance?
    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// Initializes the distance function near the free boundary.
///
/// Splits every square into two triangles. Based on the level set function with values `u` given
/// on a regular grid, it computes the distance from the _zero_ level set in the nodes of the
/// triangle through which the level set passes.  Stores the minimal value of the distance in the
/// preallocated slice `d`.
///
/// Nodes away from the boundary have their value set to `std::f64::MAX`.
///
/// `dual_norm` is the __dual__ norm. It must be an __even__ positively one-homogeneous function,
/// zero only at the origin.
pub fn init_dist_2d<F>(d: &mut [f64], u: &[f64], dim: (usize, usize), mut dual_norm: F)
where
    F: FnMut([f64; 2]) -> f64,
{
    let (nx, ny) = dim;
    assert_eq!(nx * ny, u.len());
    assert_eq!(nx * ny, d.len());

    for d in &mut *d {
        *d = std::f64::MAX;
    }

    for j in 1..nx {
        for i in 1..ny {
            let s = j * ny + i;
            let v = [s - ny - 1, s - ny, s];
            let r = triangle_dist([u[v[0]], u[v[1]], u[v[2]]], [1, 0], &mut dual_norm);
            if let Some(e) = r {
                for i in 0..3 {
                    d[v[i]] = min(e[i], d[v[i]]);
                }
            }
            let v = [s - ny - 1, s - 1, s];
            let r = triangle_dist([u[v[0]], u[v[1]], u[v[2]]], [0, 1], &mut dual_norm);
            if let Some(e) = r {
                for i in 0..3 {
                    d[v[i]] = min(e[i], d[v[i]]);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::norm::{DualNorm, EuclideanNorm};

    #[test]
    fn simple_triangles() {
        let eucl_triangle_dist = |v| triangle_dist(v, [0, 1], |p| EuclideanNorm.dual_norm(p));
        assert_eq!(eucl_triangle_dist([0., 0., 0.]), Some([0., 0., 0.]));
        assert_eq!(eucl_triangle_dist([1., 1., 1.]), None);
        assert_eq!(eucl_triangle_dist([-1., -1., -1.]), None);
        assert_eq!(
            eucl_triangle_dist([0., 1., 0.]),
            Some([0., 1. / (2f64).sqrt(), 0.])
        );
        assert_eq!(eucl_triangle_dist([0., -1., -1.]), Some([0., 1., 1.]));
        assert_eq!(eucl_triangle_dist([0., 1., 1.]), Some([0., 1., 1.]));
        assert_eq!(eucl_triangle_dist([1., 1., 0.]), Some([1., 1., 0.]));
        assert_eq!(eucl_triangle_dist([-1., 0., 0.]), Some([1., 0., 0.]));
        assert_eq!(
            eucl_triangle_dist([1., 0., 1.]),
            Some([1. / (2f64).sqrt(), 0., 1. / (2f64).sqrt()])
        );
    }

    #[test]
    fn anisotropic_norm_2d() {
        // Du = (1, 0)
        let u = [0., 0., 1., 1.];
        let mut d = [0.; 4];
        init_dist_2d(&mut d, &u, (2, 2), |p| p[0].abs().max(2. * p[1].abs()));

        assert_eq!(d, [0., 0., 1., 1.]);

        // Du = (0, 1)
        let u = [0., 1., 0., 1.];
        let mut d = [0.; 4];
        init_dist_2d(&mut d, &u, (2, 2), |p| p[0].abs().max(2. * p[1].abs()));

        assert_eq!(d, [0., 0.5, 0., 0.5]);
    }

    #[test]
    fn anisotropic_norm_3d() {
        // Du = (1, 0, 0)
        let u = [0., 0., 0., 0., 1., 1., 1., 1.];
        let mut d = [0.; 8];
        init_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0., 0., 0., 1., 1., 1., 1.]);

        // Du = (0, 1, 0)
        let u = [0., 0., 1., 1., 0., 0., 1., 1.];
        let mut d = [0.; 8];
        init_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0., 0.5, 0.5, 0., 0., 0.5, 0.5]);

        // Du = (0, 0, 1)
        let u = [0., 1., 0., 1., 0., 1., 0., 1.];
        let mut d = [0.; 8];
        init_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25]);
    }
}
