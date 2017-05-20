extern crate fast_sweeping;
extern crate gnuplot;

#[allow(unused_imports)]
use gnuplot::{Figure, Caption, Color, Fix, AxesCommon, PlotOption, DashType, Coordinate,
              TextColor, AutoOption, ContourStyle};
use fast_sweeping::signed_distance_2d;
use fast_sweeping::anisotropic_signed_distance_2d;

fn main() {
    let n = 64;
    let mut u = vec![0.; (n + 1) * (n + 1)];

    let h = 1f64 / n as f64;

    let r = 0.3;

    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            let x = i as f64 * h - 0.5;
            let y = j as f64 * h - 0.5;
            u[i + j * (n + 1)] = (x * x + y * y).sqrt() - r;
            // u[i + j * (n + 1)] = (x.abs()).max(y.abs()) - r;
        }
    }

    let orig = u.clone();

    let mut d = vec![0f64; (n + 1) * (n + 1)];

    // signed_distance_2d(&mut d, &u, (n + 1, n + 1), h);
    // anisotropic_signed_distance_2d(&mut d, &u, (n + 1, n + 1), h, |p| {
    //     p[0].abs().max(p[1].abs())
    // }, |d, v, _| {
    //     d.min(v[0].min(v[1]) + 1.)
    // });
    anisotropic_signed_distance_2d(&mut d,
                                   &u,
                                   (n + 1, n + 1),
                                   h,
                                   |p| p[0].abs() + p[1].abs(),
                                   |d, v, _| {
                                       d.min(v[0] + 1.)
                                           .min(v[1] + 1.)
                                           .min(0.5 * (v[0] + v[1] + 1.))
                                   });

    u.clone_from(&d);

    let mut err = d.clone();
    for (err, u) in err.iter_mut().zip(orig.iter()) {
        *err -= *u;
    }

    let mut fg = Figure::new();

    fg.axes3d()
        .set_view_map()
        .set_aspect_ratio(AutoOption::Fix(1.))
        // .surface(&err, n+1, n+1, None, &[])
        .surface(&d, n+1, n+1, None, &[])
        .show_contours(false, true, ContourStyle::Linear, AutoOption::Fix(""), AutoOption::Fix(20))
        // .surface(&u, n+1, n+1, None, &[])
        ;
    fg.show();
}
