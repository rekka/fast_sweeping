//! Implementations of the Hausdorff distance function between level sets.
use isosurface::{marching_tetrahedra_with_data_emit, marching_triangles_with_data_emit};
use signed_distance_2d;
use signed_distance_3d;

/// Computes the Hausdorff distance function between two level sets of discrete functions given on
/// the same square grid.
pub fn hausdorff_dist_2d(u: &[f64], v: &[f64], dim: (usize, usize), h: f64) -> f64 {
    assert_eq!(u.len(), dim.0 * dim.1);
    assert_eq!(v.len(), dim.0 * dim.1);

    let mut m: f64 = 0.;

    // compute the max of the distance function on both level sets
    let mut dist = vec![0.; u.len()];
    signed_distance_2d(&mut dist, v, dim, h);
    marching_triangles_with_data_emit(u, &dist, dim, 0., |_, d| {
        m = m.max(d[0].abs()).max(d[1].abs());
    });

    signed_distance_2d(&mut dist, u, dim, h);
    marching_triangles_with_data_emit(v, &dist, dim, 0., |_, d| {
        m = m.max(d[0].abs()).max(d[1].abs());
    });

    m
}

/// Computes the Hausdorff distance function between two level sets of discrete functions given on
/// the same square grid.
pub fn hausdorff_dist_3d(u: &[f64], v: &[f64], dim: (usize, usize, usize), h: f64) -> f64 {
    assert_eq!(u.len(), dim.0 * dim.1 * dim.2);
    assert_eq!(v.len(), dim.0 * dim.1 * dim.2);

    let mut m: f64 = 0.;

    // compute the max of the distance function on both level sets
    let mut dist = vec![0.; u.len()];
    signed_distance_3d(&mut dist, v, dim, h);
    marching_tetrahedra_with_data_emit(u, &dist, dim, 0., |_, d| {
        m = m.max(d[0].abs()).max(d[1].abs()).max(d[2].abs());
    });

    signed_distance_3d(&mut dist, u, dim, h);
    marching_tetrahedra_with_data_emit(v, &dist, dim, 0., |_, d| {
        m = m.max(d[0].abs()).max(d[1].abs()).max(d[2].abs());
    });

    m
}

/// Line integral of the square of a linear function.
fn line_integral_sq(u: f64, v: f64) -> f64 {
    (1. / 3.) * (u * u + u * v + v * v)
}

/// Computes a L² version of the Hausdorff distance by integrating the square
/// of the distance functions over the level set.
///
/// Returns +∞ if at least one of the level sets is empty.
pub fn l2_hausdorff_dist_2d(u: &[f64], v: &[f64], dim: (usize, usize), h: f64) -> f64 {
    assert_eq!(u.len(), dim.0 * dim.1);
    assert_eq!(v.len(), dim.0 * dim.1);

    let mut i: f64 = 0.;

    // compute the line integrals of the square of the distance function

    let mut dist = vec![0.; u.len()];
    signed_distance_2d(&mut dist, v, dim, h);
    marching_triangles_with_data_emit(u, &dist, dim, 0., |c, d| {
        let a = h * (c[0][0] - c[1][0]).hypot(c[0][1] - c[1][1]);
        i += a * line_integral_sq(d[0], d[1]);
    });

    signed_distance_2d(&mut dist, u, dim, h);
    marching_triangles_with_data_emit(v, &dist, dim, 0., |c, d| {
        let a = h * (c[0][0] - c[1][0]).hypot(c[0][1] - c[1][1]);
        i += a * line_integral_sq(d[0], d[1]);
    });

    i.sqrt()
}

/// Triangle integral of the square of a linear function.
fn triangle_integral_sq(u: f64, v: f64, w: f64) -> f64 {
    (1. / 6.) * (u * u + u * v + v * v + u * w + v * w + w * w)
}

/// Area of triangle in 3D.
fn triangle_area(c: [[f64; 3]; 3]) -> f64 {
    // FIXME: stable area using Heron?
    fn cross_norm(x: [f64; 3], y: [f64; 3]) -> f64 {
        let z0 = x[1] * y[2] - x[2] * y[1];
        let z1 = x[2] * y[0] - x[0] * y[2];
        let z2 = x[0] * y[1] - x[1] * y[0];

        (z0 * z0 + z1 * z1 + z2 * z2).sqrt()
    }
    fn sub(x: [f64; 3], y: [f64; 3]) -> [f64; 3] {
        [x[0] - y[0], x[1] - y[1], x[2] - y[2]]
    }
    0.5 * cross_norm(sub(c[1], c[0]), sub(c[2], c[0]))
}

/// Computes a L² version of the Hausdorff distance by integrating the square
/// of the distance functions over the level set.
///
/// Returns +∞ if at least one of the level sets is empty.
pub fn l2_hausdorff_dist_3d(u: &[f64], v: &[f64], dim: (usize, usize, usize), h: f64) -> f64 {
    assert_eq!(u.len(), dim.0 * dim.1 * dim.2);
    assert_eq!(v.len(), dim.0 * dim.1 * dim.2);

    let mut i: f64 = 0.;

    // compute the line integrals of the square of the distance function

    let mut dist = vec![0.; u.len()];
    signed_distance_3d(&mut dist, v, dim, h);
    marching_tetrahedra_with_data_emit(u, &dist, dim, 0., |c, d| {
        let a = triangle_area(c);
        i += a * triangle_integral_sq(d[0], d[1], d[2]);
    });

    signed_distance_3d(&mut dist, u, dim, h);
    marching_tetrahedra_with_data_emit(v, &dist, dim, 0., |c, d| {
        let a = triangle_area(c);
        i += a * triangle_integral_sq(d[0], d[1], d[2]);
    });

    i.sqrt() * h
}
