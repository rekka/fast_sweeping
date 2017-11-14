use signed_distance_2d;
use isosurface::marching_triangles_with_data_emit;

/// Computes the Hausdorff distance function between two level sets of discrete functions given on
/// the same square grid.
pub fn hausdorff_dist_2d(u: &[f64], v: &[f64], dim: (usize, usize), h: f64) -> f64 {
    assert_eq!(dim.0, dim.1);
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

/// Line integral of the square of a linear function.
fn line_integral_sq(u: f64, v: f64) -> f64 {
    (1. / 3.) * (u * u + u * v + v * v)
}

/// Computes a L² version of the Hausdorff distance by integrating the square
/// of the distance functions over the level set.
pub fn l2_hausdorff_dist_2d(u: &[f64], v: &[f64], dim: (usize, usize), h: f64) -> f64 {
    assert_eq!(dim.0, dim.1);
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
