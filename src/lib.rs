/// Compute the signed distance function from a line segment given as the level set of a linear
/// function on an isosceles right-angle triangle.
///
/// Input are `u`, the values at the verteces. The vertex 0 is the one with the right angle.
///
/// `h` is the length of the side.
///
/// The return is the values of the signed distance function or `None` if there line segment does
/// not pass through the triangle.
pub fn triangle_dist(mut u: [f64; 3]) -> Option<[f64; 3]> {
    // normalize so that u[0] >= 0.
    if u[0] < 0. {
        for i in 0..3 {
            u[i] = -u[i];
        }
    }

    // gradient vector
    let gx = u[1] - u[0];
    let gy = u[2] - u[0];

    let g_norm = (gx * gx + gy * gy).sqrt();

    if u[1] >= 0. {
        if u[2] >= 0. {
            return None;
        } else { // u[2] < 0.
            // intersect position
            let i02 = u[0] / (u[0] - u[2]);
            let i12 = (2f64).sqrt() * u[1] / (u[1] - u[2]);
            // find the direction of the gradient
            // to deduce the vertex that is closest to the line
            if gx <= 0. { // 0
                return Some([u[0] / g_norm, i12, 1. - i02]);
            } else if gx > -gy { // 1
                return Some([i02, u[1] / g_norm, (2f64).sqrt() - i12]);
            } else { // 2
                return Some([i02, i12, -u[2] / g_norm]);
            }
        }
    } else { // u[1] < 0.
        if u[2] >= 0. {
            // intersect position
            let i01 = u[0] / (u[0] - u[1]);
            let i12 = (2f64).sqrt() * u[1] / (u[1] - u[2]);
            // find the direction of the gradient
            // to deduce the vertex that is closest to the line
            if gy <= 0. { // 0
                return Some([u[0] / g_norm, 1. - i01, (2f64).sqrt() - i12]);
            } else if -gx > gy { // 1
                return Some([i01, -u[1] / g_norm, (2f64).sqrt() - i12]);
            } else { // 2
                return Some([i01, i12, u[2] / g_norm]);
            }
        } else { // u[2] < 0.
            // intersect position
            let i10 = u[1] / (u[1] - u[0]);
            let i20 = u[2] / (u[2] - u[0]);

            return Some([u[0] / g_norm, i10, i20]);
        }
    }

    Some([0., 0., 0.])
}

/// Initialize distance around the free boundary.
///
/// Splits every square into two triangles and computes the distance on each of them.
pub fn init_dist(d: &mut [f64], u: &[f64], dim: (usize, usize)) {
    assert_eq!(dim.0 * dim.1, u.len());
    assert_eq!(dim.0 * dim.1, d.len());

    for i in 0..d.len() {
        d[i] = std::f64::MAX;
    }

    for j in 1..dim.1 {
        for i in 1..dim.0 {
            let s = j * dim.0 + i;
            let r = triangle_dist([u[s - dim.0 - 1], u[s - dim.0], u[s - 1]]);
            if let Some(e) = r {
                d[s - dim.0 - 1] = e[0].min(d[s - dim.0 - 1]);
                d[s - dim.0] = e[1].min(d[s - dim.0]);
                d[s - 1] = e[2].min(d[s - 1]);
            }
            let r = triangle_dist([u[s], u[s - dim.0], u[s - 1]]);
            if let Some(e) = r {
                d[s] = e[0].min(d[s]);
                d[s - dim.0] = e[1].min(d[s - dim.0]);
                d[s - 1] = e[2].min(d[s - 1]);
            }
        }
    }

    // for i in 0..d.len() {
    //     if u[i] < 0. {
    //         d[i] = -d[i];
    //     }
    // }
}

/// Computes the signed distance function in 2D using the Fast Sweeping algorithm.
pub fn fast_sweep_dist(d: &mut [f64], dim: (usize, usize)) {

    for k in 1..5 {
        for q in 0..dim.1 {
            let j = if k == 3 || k == 4 {
                dim.1 - 1 - q
            } else {
                q
            };
            for p in 0..dim.0 {
                let i = if k == 2 || k == 3 {
                    dim.0 - 1 - p
                } else {
                    p
                };
                let s = j * dim.0 + i;
                let a = if i == 0 {
                    d[s + 1]
                } else if i == dim.0 - 1 {
                    d[s - 1]
                } else {
                    d[s - 1].min(d[s + 1])
                };
                let b = if j == 0 {
                    d[s + dim.0]
                } else if j == dim.1 - 1 {
                    d[s - dim.0]
                } else {
                    d[s - dim.0].min(d[s + dim.0])
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

#[cfg(test)]
mod test {
    use super::*;
    extern crate quickcheck;
    use self::quickcheck::quickcheck;

    #[test]
    fn it_works() {
        let n = 8;
        let mut u = vec![0.; (n + 1) * (n + 1)];

        let h = 1f64;

        let gx = 0.5f64;
        let gy = 0.4f64;
        let g_norm = (gx * gx + gy * gy).sqrt();
        let gx = gx / g_norm;
        let gy = gy / g_norm;
        let c = -2.;

        for i in 0..(n+1) {
            for j in 0..(n+1) {
                let x = i as f64 * h;
                let y = j as f64 * h;
                u[i + j * (n + 1)] = x * gx + y * gy + c;
            }
        }

        let mut d = vec![0f64; (n + 1) * (n + 1)];

        init_dist(&mut d, &u, (n+1, n+1));
        fast_sweep_dist(&mut d, (n+1, n+1));

        for i in 0..d.len() {
            if u[i] < 0. {
                d[i] = -d[i];
            }
        }
        for (d, u) in d.iter_mut().zip(u.iter()) {
            *d -= *u;
        }
        println!("");
        println!("{:?}", u);
        println!("{:?}", d);
        panic!();
    }

    #[test]
    fn it_is_nonnegative() {
        fn prop(x: f64, y: f64, z: f64) -> bool {
            if let Some(e) = triangle_dist([x, y, z]) {
                // println!("{:?} -> {:?}", (x, y, z), e);
                e.into_iter().all(|&x| x >= 0.)
            } else {
                true
            }
        }
        quickcheck(prop as fn(f64, f64, f64) -> bool);
    }
}
