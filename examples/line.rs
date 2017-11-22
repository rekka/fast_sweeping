extern crate fast_sweeping;
extern crate gnuplot;

#[allow(unused_imports)]
use gnuplot::{AxesCommon, Caption, Color, Coordinate, DashType, Figure, Fix, PlotOption, TextColor};
use fast_sweeping::signed_distance_2d;

fn main() {
    let n = 64;
    let mut u = vec![0.; (n + 1) * (n + 1)];

    let h = 1f64 / n as f64;

    let ta = 1.3f64;
    let gx = ta.cos();
    let gy = ta.sin();

    let c = -(gx + gy) * n as f64 * 0.5 * h;

    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            let x = i as f64 * h;
            let y = j as f64 * h;
            u[i + j * (n + 1)] = x * gx + y * gy + c;
        }
    }

    let mut d = vec![0f64; (n + 1) * (n + 1)];

    signed_distance_2d(&mut d, &u, (n + 1, n + 1), h);

    let mut err = d.clone();
    for (err, u) in err.iter_mut().zip(u.iter()) {
        *err -= *u;
    }

    let mut fg = Figure::new();

    fg.axes3d()
        .surface(&err, n+1, n+1, None, &[])
        // .surface(&d, n+1, n+1, None, &[])
        .surface(&u, n+1, n+1, None, &[]);
    fg.show();
}
