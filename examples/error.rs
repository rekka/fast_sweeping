//! Visualizes the error of fast sweeping.
extern crate docopt;
extern crate fast_sweeping;
extern crate gnuplot;
extern crate isosurface;
#[macro_use(s)]
extern crate ndarray;
extern crate rustc_serialize;

use ndarray::prelude::*;
use ndarray::Data;
use ndarray::Zip;
use fast_sweeping::signed_distance_2d;
#[allow(unused_imports)]
use gnuplot::{AutoOption, AxesCommon, Caption, Color, ContourStyle, Coordinate, DashType, Figure,
              Fix, PlotOption, TextColor};
use std::f64::NAN;

const USAGE: &'static str = "
Show effect of redistance.

Usage:
  error [options]
  error (-h | --help)
  error --version

Options:
  -n INT                Mesh  resolution (n^2). [default: 16]
  --delta FLOAT         Size of neighborhood rel. to 1 / n [default: 1.5]
  --svg FILE            Produce svg output to FILE.
  -h, --help            Show this screen.
  --version             Show version.
";

#[derive(Debug, RustcDecodable)]
pub struct Args {
    flag_n: usize,
    flag_svg: Option<String>,
    flag_delta: f64,
}

fn tensor_product<A, B, C, S, T, F>(x: &ArrayBase<S, Ix1>, y: &ArrayBase<T, Ix1>, f: F) -> Array2<C>
where
    S: Data<Elem = A>,
    T: Data<Elem = B>,
    A: Copy,
    B: Copy,
    F: Fn(A, B) -> C,
{
    let dim = (x.len(), y.len());
    let mut r = Vec::with_capacity(dim.0 * dim.1);
    for i in 0..dim.0 {
        for j in 0..dim.1 {
            r.push(f(x[i], y[j]));
        }
    }

    Array::from_shape_vec(dim, r).unwrap()
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());


    let n = args.flag_n;
    let h = 1. / n as f64;

    let (e, ge) = error(n, args.flag_delta);

    let m = e.fold(0f64, |m, &x| m.max(x));
    let gm = ge.fold(0f64, |m, &x| m.max(x));

    println!("max error      = {:.4} * h²", m / h.powi(2));
    println!("max grad error = {:.4} * h", gm / h);

    let mut fg = Figure::new();
    if let Some(f) = args.flag_svg {
        fg.set_terminal("svg size 1280, 1280", &f);
    }
    fg.axes3d()
        .set_title("Max gradient error", &[])
        .set_view_map()
        .set_aspect_ratio(AutoOption::Fix(1.))
        .surface(
            ge.iter(),
            ge.dim().0,
            ge.dim().1,
            Some((-0.5, -0.5, 0.5, 0.5)),
            &[],
        );
    fg.show();
}

fn error(n: usize, delta: f64) -> (Array2<f64>, Array2<f64>) {
    let h = 1. / n as f64;
    let delta = delta * h;

    let r = 0.3;

    let xs: Array1<f64> = Array::linspace(-0.5, 0.5, n + 1);
    let ys: Array1<f64> = Array::linspace(-0.5, 0.5, n + 1);
    let u = tensor_product(&xs, &ys, |x, y| x.hypot(y) - r);
    let gu = tensor_product(&xs, &ys, |x, y| {
        let norm = x.hypot(y);
        if norm > 0. {
            (x / norm, y / norm)
        } else {
            (NAN, NAN)
        }
    });

    // initial data
    let mut d = u.clone();

    signed_distance_2d(d.as_slice_mut().unwrap(), u.as_slice().unwrap(), u.dim(), h);

    let mut diff = u.clone();

    Zip::from(&mut diff).and(&d).apply(|diff, &d| {
        *diff = if diff.abs() > delta {
            NAN
        } else {
            (*diff - d).abs()
        }
    });

    let mut gdiff = u.slice(s![1..-1, 1..-1]).to_owned();

    // gradient via the central difference
    Zip::from(&mut gdiff)
        .and(d.slice(s![..-2, 1..-1]))
        .and(d.slice(s![2.., 1..-1]))
        .and(d.slice(s![1..-1, ..-2]))
        .and(d.slice(s![1..-1, 2..]))
        .and(gu.slice(s![1..-1, 1..-1]))
        .apply(|gdiff, &xm, &xp, &ym, &yp, &gu| {
            let dx = (xp - xm) / (2. * h) - gu.0;
            let dy = (yp - ym) / (2. * h) - gu.1;

            *gdiff = if gdiff.abs() > delta {
                NAN
            } else {
                dx.hypot(dy)
            }
        });

    (diff, gdiff)
}
