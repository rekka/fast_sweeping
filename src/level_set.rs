use std;
use super::min;

/// Computes the signed distance function from a plane given as the _zero_ level set of a
/// linear function on a tetrahedron at 4 points with unit coordinates starting at (0, 0, 0) and
/// ending at (1, 1, 1), and in between exactly one coordinate changes from 0 to 1.
///
/// Inputs are `u`, the values at the vertices.
///
/// The function returns the values of the (non-signed) distance function or `None` if the zero
/// level set does not pass through the tetrahedron.
pub fn tetrahedron_anisotropic_dist<F>(
    mut u: [f64; 4],
    mut dual_norm: F,
    perm: [usize; 3],
) -> Option<[f64; 4]>
where
    F: FnMut([f64; 3]) -> f64,
{
    let mut n_pos = 0;
    let mut n_neg = 0;
    for u in &mut u {
        if *u > 0. {
            n_pos += 1;
        } else if *u < 0. {
            n_neg += 1;
        }
    }
    // check if sign differs (level set goes throught the triangle)
    if n_neg == 4 || n_pos == 4 {
        return None;
    }

    // everything is zero
    if n_neg + n_pos == 0 {
        return Some([0.; 4]);
    }

    let g = [u[1] - u[0], u[2] - u[1], u[3] - u[2]];
    let g = [g[perm[0]], g[perm[1]], g[perm[2]]];
    let g_norm_rcp = 1. / dual_norm(g);

    // FIXME: this requires for norm to be even
    // Support triangular anisotropies?
    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// Initializes the distance around the free boundary.
///
/// Based on the level set function with values `u` given on a regular grid, it computes the
/// distance from the _zero_ level set in the nodes of the triangles through which the level set
/// passes.  Stores the result in the preallocated slice `d`.
///
/// Nodes away from the boundary have their value set to `std::f64::MAX`.
///
/// Splits every cube into six tetrahedra and computes the distance on each of them.
pub fn init_anisotropic_dist_3d<F>(
    d: &mut [f64],
    u: &[f64],
    dim: (usize, usize, usize),
    mut dual_norm: F,
) where
    F: FnMut([f64; 3]) -> f64,
{
    let (nx, ny, nz) = dim;
    assert_eq!(nx * ny * nz, u.len());
    assert_eq!(nx * ny * nz, d.len());

    for d in &mut *d {
        *d = std::f64::MAX;
    }

    // split each cube into 6 tetrahedrons
    let ids = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)],
    ];

    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [2, 0, 1],
        [1, 2, 0],
        [2, 1, 0],
    ];

    for i in 1..nx {
        for j in 1..ny {
            for k in 1..nz {
                let s = i * ny * nz + j * nz + k;
                let mut v = [0.; 4];

                for (idx, perm) in ids.iter().zip(perms.iter()) {
                    for m in 0..4 {
                        v[m] = u[s - idx[m].0 * ny * nz - idx[m].1 * nz - idx[m].2];
                    }

                    let r = tetrahedron_anisotropic_dist(v, &mut dual_norm, *perm);
                    if let Some(r) = r {
                        for m in 0..4 {
                            let q = s - idx[m].0 * ny * nz - idx[m].1 * nz - idx[m].2;
                            d[q] = min(d[q], r[m]);
                        }
                    }
                }
            }
        }
    }
}

/// Computes the signed distance function from a plane given as the _zero_ level set of a
/// linear function on a tetrahedron at 4 points with unit coordinates starting at (0, 0, 0) and
/// ending at (1, 1, 1), and in between exactly one coordinate changes from 0 to 1.
///
/// Inputs are `u`, the values at the vertices.
///
/// The function returns the values of the (non-signed) distance function or `None` if the zero
/// level set does not pass through the tetrahedron.
pub fn tetrahedron_dist(mut u: [f64; 4]) -> Option<[f64; 4]> {
    let mut n_pos = 0;
    let mut n_neg = 0;
    for u in &mut u {
        if *u > 0. {
            n_pos += 1;
        } else if *u < 0. {
            n_neg += 1;
        }
    }
    // check if sign differs (level set goes throught the triangle)
    if n_neg == 4 || n_pos == 4 {
        return None;
    }

    // everything is zero
    if n_neg + n_pos == 0 {
        return Some([0.; 4]);
    }

    let g_norm_rcp = 1.
        / u.windows(2)
            .fold(0., |sum, x| sum + (x[1] - x[0]).powi(2))
            .sqrt();

    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// Initializes the distance around the free boundary.
///
/// Based on the level set function with values `u` given on a regular grid, it computes the
/// distance from the _zero_ level set in the nodes of the triangles through which the level set
/// passes.  Stores the result in the preallocated slice `d`.
///
/// Nodes away from the boundary have their value set to `std::f64::MAX`.
///
/// Splits every square into two triangles and computes the distance on each of them.
pub fn init_dist_3d(d: &mut [f64], u: &[f64], dim: (usize, usize, usize)) {
    let (nx, ny, nz) = dim;
    assert_eq!(nx * ny * nz, u.len());
    assert_eq!(nx * ny * nz, d.len());

    for d in &mut *d {
        *d = std::f64::MAX;
    }

    // split each cube into 6 tetrahedrons
    let ids = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)],
    ];

    for i in 1..nx {
        for j in 1..ny {
            for k in 1..nz {
                let s = i * ny * nz + j * nz + k;
                let mut v = [0.; 4];

                for idx in ids.iter() {
                    for m in 0..4 {
                        v[m] = u[s - idx[m].0 * ny * nz - idx[m].1 * nz - idx[m].2];
                    }

                    let r = tetrahedron_dist(v);
                    if let Some(r) = r {
                        for m in 0..4 {
                            let q = s - idx[m].0 * ny * nz - idx[m].1 * nz - idx[m].2;
                            d[q] = min(d[q], r[m]);
                        }
                    }
                }
            }
        }
    }
}

/// Computes the signed distance function from a line given as the _zero_ level set of a
/// linear function on an isosceles right triangle. It preserves level sets of linear functions.
///
/// There does not see to be much point of trying to be smart and compute the distance to the line
/// segments within the triangle. This shifts the level set even in a symmetric case when not
/// needed. This simpler approach here seems to better preserve corners in the crystalline mean
/// curvature flow.
///
/// Inputs are the values at the vertices. The vertex 0 is the one with the right angle.
///
/// The function returns the values of the distance function or `None` if the zero level set
/// does not pass through the triangle.
pub fn triangle_dist(mut u: [f64; 3]) -> Option<[f64; 3]> {
    let mut n_pos = 0;
    let mut n_neg = 0;
    for u in &mut u {
        if *u > 0. {
            n_pos += 1;
        } else if *u < 0. {
            n_neg += 1;
        }
    }
    // check if sign differs (level set goes throught the triangle)
    if n_neg == 3 || n_pos == 3 {
        return None;
    }

    // everything is zero
    if n_neg + n_pos == 0 {
        return Some([0., 0., 0.]);
    }

    let gx = u[1] - u[0];
    let gy = u[2] - u[0];
    let g_norm_rcp = 1. / (gx * gx + gy * gy).sqrt();

    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// Initializes the distance around the free boundary.
///
/// Based on the _linear_ level set function with values `u` given on a regular grid, it computes
/// the distance from the _zero_ level set (a line) in the nodes of the triangles through which the
/// level set passes.  Stores the result in the preallocated slice `d`.
///
/// Nodes away from the boundary have their value set to `std::f64::MAX`.
///
/// Splits every square into two triangles and computes the distance on each of them.
///
/// ```text,ignore
///   0,1 *----* 1,1
///       |   /|
///       |  / |
///       | /  |
///       |/   |
///   0,0 *----* 1,0
/// ```
pub fn init_dist_2d(d: &mut [f64], u: &[f64], dim: (usize, usize)) {
    let (nx, ny) = dim;
    assert_eq!(nx * ny, u.len());
    assert_eq!(nx * ny, d.len());

    for d in &mut *d {
        *d = std::f64::MAX;
    }

    for j in 1..nx {
        for i in 1..ny {
            let s = j * ny + i;
            let vs = [[s - ny, s - ny - 1, s], [s - 1, s - ny - 1, s]];
            for v in &vs {
                let r = triangle_dist([u[v[0]], u[v[1]], u[v[2]]]);
                if let Some(e) = r {
                    for i in 0..3 {
                        d[v[i]] = min(e[i], d[v[i]]);
                    }
                }
            }
        }
    }
}

fn triangle_anisotropic_dist<F>(
    mut u: [f64; 3],
    perm: [usize; 2],
    mut dual_norm: F,
) -> Option<[f64; 3]>
where
    F: FnMut([f64; 2]) -> f64,
{
    let mut n_pos = 0;
    let mut n_neg = 0;
    for u in &mut u {
        if *u > 0. {
            n_pos += 1;
        } else if *u < 0. {
            n_neg += 1;
        }
    }
    // check if sign differs (level set goes throught the triangle)
    if n_neg == 3 || n_pos == 3 {
        return None;
    }

    // everything is zero
    if n_neg + n_pos == 0 {
        return Some([0., 0., 0.]);
    }

    let g = [u[1] - u[0], u[2] - u[1]];
    let g = [g[perm[0]], g[perm[1]]];
    let g_norm_rcp = 1. / dual_norm(g);

    for u in u.iter_mut() {
        *u = u.abs() * g_norm_rcp;
    }
    Some(u)
}

/// As `init_dist_2d`, but for general anisotropic norm.
///
/// `dual_norm` is the __dual__ norm. It must be a positively one-homogeneous function.
pub fn init_anisotropic_dist_2d<F>(d: &mut [f64], u: &[f64], dim: (usize, usize), mut dual_norm: F)
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
            let r = triangle_anisotropic_dist([u[v[0]], u[v[1]], u[v[2]]], [1, 0], &mut dual_norm);
            if let Some(e) = r {
                for i in 0..3 {
                    d[v[i]] = min(e[i], d[v[i]]);
                }
            }
            let v = [s - ny - 1, s - 1, s];
            let r = triangle_anisotropic_dist([u[v[0]], u[v[1]], u[v[2]]], [0, 1], &mut dual_norm);
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

    #[test]
    fn simple_triangles() {
        assert_eq!(triangle_dist([0., 0., 0.]), Some([0., 0., 0.]));
        assert_eq!(triangle_dist([1., 1., 1.]), None);
        assert_eq!(triangle_dist([-1., -1., -1.]), None);
        assert_eq!(triangle_dist([0., 1., 0.]), Some([0., 1., 0.]));
        assert_eq!(
            triangle_dist([0., -1., -1.]),
            Some([0., 1. / (2f64).sqrt(), 1. / (2f64).sqrt()])
        );
        assert_eq!(
            triangle_dist([0., 1., 1.]),
            Some([0., 1. / (2f64).sqrt(), 1. / (2f64).sqrt()])
        );
        assert_eq!(triangle_dist([1., 1., 0.]), Some([1., 1., 0.]));
        assert_eq!(
            triangle_dist([-1., 0., 0.]),
            Some([1. / (2f64).sqrt(), 0., 0.])
        );
    }

    #[test]
    fn anisotropic_norm_2d() {
        // Du = (1, 0)
        let u = [0., 0., 1., 1.];
        let mut d = [0.; 4];
        init_anisotropic_dist_2d(&mut d, &u, (2, 2), |p| p[0].abs().max(2. * p[1].abs()));

        assert_eq!(d, [0., 0., 1., 1.]);

        // Du = (0, 1)
        let u = [0., 1., 0., 1.];
        let mut d = [0.; 4];
        init_anisotropic_dist_2d(&mut d, &u, (2, 2), |p| p[0].abs().max(2. * p[1].abs()));

        assert_eq!(d, [0., 0.5, 0., 0.5]);
    }

    #[test]
    fn anisotropic_norm_3d() {
        // Du = (1, 0, 0)
        let u = [0., 0., 0., 0., 1., 1., 1., 1.];
        let mut d = [0.; 8];
        init_anisotropic_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0., 0., 0., 1., 1., 1., 1.]);

        // Du = (0, 1, 0)
        let u = [0., 0., 1., 1., 0., 0., 1., 1.];
        let mut d = [0.; 8];
        init_anisotropic_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0., 0.5, 0.5, 0., 0., 0.5, 0.5]);

        // Du = (0, 0, 1)
        let u = [0., 1., 0., 1., 0., 1., 0., 1.];
        let mut d = [0.; 8];
        init_anisotropic_dist_3d(&mut d, &u, (2, 2, 2), |p| {
            p[0].abs().max(2. * p[1].abs()).max(4. * p[2].abs())
        });

        assert_eq!(d, [0., 0.25, 0., 0.25, 0., 0.25, 0., 0.25]);
    }
}
