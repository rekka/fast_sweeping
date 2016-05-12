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
