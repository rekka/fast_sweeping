extern crate fast_sweeping;

use fast_sweeping::dist::hausdorff_dist_2d;
use fast_sweeping::dist::l2_hausdorff_dist_2d;

fn main() {
    let n = 256;
    let mut u = vec![0.; (n + 1) * (n + 1)];
    let mut v = vec![0.; (n + 1) * (n + 1)];

    let h = 1f64 / n as f64;

    let r = 0.3;
    let delta = 0.1;

    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            let x = i as f64 * h - 0.5;
            let y = j as f64 * h - 0.5;
            u[i + j * (n + 1)] = x.hypot(y) - r;
            v[i + j * (n + 1)] = x.hypot(y) - r + delta;
            // u[i + j * (n + 1)] = x.abs().max(y.abs()) - r;
            // v[i + j * (n + 1)] = x.abs().max(y.abs()) - r + 0.1;
            // u[i + j * (n + 1)] = x + 0.2 * y;
            // v[i + j * (n + 1)] = x + 0.2 * y + 0.2;
            // u[i + j * (n + 1)] = (x.abs()).max(y.abs()) - r;
        }
    }

    let d = hausdorff_dist_2d(&u, &v, (n + 1, n + 1), 1. / n as f64);
    let d_l2 = l2_hausdorff_dist_2d(&u, &v, (n + 1, n + 1), 1. / n as f64);

    println!("Hausdorff distance = {}, expected = {}", d, 0.1);
    println!(
        "L²-Hausdorff distance = {}, expected = {}",
        d_l2,
        (2. * std::f64::consts::PI * (2. * r - delta)).sqrt() * delta
    );
}
