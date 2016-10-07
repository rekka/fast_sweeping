//! Illustrates the effect of an iterative application of the signed distance computation to a
//! circle.
extern crate fast_sweeping;
extern crate ndarray;
extern crate gnuplot;
extern crate pbr;

use ndarray::prelude::*;
use ndarray::Data;
use fast_sweeping::signed_distance_2d;
#[allow(unused_imports)]
use gnuplot::{Figure, Caption, Color, Fix, AxesCommon, PlotOption, DashType, Coordinate,
              TextColor, ContourStyle, AutoOption};
use pbr::ProgressBar;

fn tensor_product<A, B, C, S, T, F>(x: &ArrayBase<S, Ix>,
                                    y: &ArrayBase<T, Ix>,
                                    f: F)
                                    -> Array<C, (Ix, Ix)>
    where S: Data<Elem = A>,
          T: Data<Elem = B>,
          A: Copy,
          B: Copy,
          F: Fn(A, B) -> C
{
    let dim = (x.len(), y.len());
    let mut r = Vec::with_capacity(dim.0 * dim.1);
    for j in 0..dim.1 {
        for i in 0..dim.0 {
            r.push(f(x[i], y[j]));
        }
    }

    Array::from_shape_vec(dim, r).unwrap()
}

fn main() {
    let n = 8;
    let dim = (n + 1, n + 1);

    let h = 1. / n as f64;

    let k = 128;

    let r = 0.3;

    let xs: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let ys: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let mut u = tensor_product(&xs, &ys, |x, y| (x * x + y * y).sqrt() - r);

    // initial data
    let mut d = u.clone();

    let mut frame_counter = 0;
    let mut fg = Figure::new();

    println!("Saving output to /tmp/plot####.png");
    // repeatedly compute the signed distance function
    let mut pb = ProgressBar::new(k);
    for _ in 0..k {
        pb.inc();
        // compute the distance function
        signed_distance_2d(d.as_slice_mut().unwrap(), u.as_slice().unwrap(), dim, h);
        u.clone_from(&d);

        {
            fg.clear_axes();

            fg.set_terminal("pngcairo size 1920, 1080",
                            &format!("/tmp/plot{:04}.png", frame_counter));
            fg.axes3d()
              .set_aspect_ratio(AutoOption::Fix(1.))
              .set_view_map()
              .show_contours_custom(true, true, ContourStyle::Linear, AutoOption::Auto, &[0.])
              .surface(&u, n + 1, n + 1, None, &[]);
            fg.show();
            frame_counter += 1;
        }
    }
}
