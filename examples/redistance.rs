//! Illustrates the effect of an iterative application of the signed distance computation to a
//! curve.
extern crate docopt;
extern crate fast_sweeping;
extern crate gnuplot;
extern crate isosurface;
extern crate ndarray;
extern crate rustc_serialize;

use ndarray::prelude::*;
use ndarray::Data;
use fast_sweeping::signed_distance_2d;
#[allow(unused_imports)]
use gnuplot::{AutoOption, AxesCommon, Caption, Color, ContourStyle, Coordinate, DashType, Figure,
              Fix, PlotOption, TextColor};

const USAGE: &'static str = "
Show effect of redistance.

Usage:
  redistance [options]
  redistance (-h | --help)
  redistance --version

Options:
  -n INT                Mesh  resolution (n^2). [default: 8]
  --svg FILE            Produce svg output to FILE.
  -h, --help            Show this screen.
  --version             Show version.
";

#[derive(Debug, RustcDecodable)]
pub struct Args {
    flag_n: usize,
    flag_svg: Option<String>,
}

fn tensor_product<A, B, C, S, T, F>(
    x: &ArrayBase<S, Ix1>,
    y: &ArrayBase<T, Ix1>,
    f: F,
) -> Array<C, Ix2>
where
    S: Data<Elem = A>,
    T: Data<Elem = B>,
    A: Copy,
    B: Copy,
    F: Fn(A, B) -> C,
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
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.decode())
        .unwrap_or_else(|e| e.exit());

    let n = args.flag_n;
    let dim = (n + 1, n + 1);

    let h = 1. / n as f64;

    let k = 1;
    let k1 = 10 - k;

    let r = 0.3;

    let xs: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let ys: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let mut u = tensor_product(&xs, &ys, |x, y| {
        (x * x + y * y).sqrt() - r + 0.2 * (x * 12.).sin()
    });
    // let mut u = tensor_product(&xs, &ys, |x, y| (x * x + y * y).sqrt() - r);

    // initial data
    let mut d = u.clone();

    let init_verts = isosurface::marching_triangles(u.as_slice().unwrap(), dim, 0.);

    for _ in 0..k {
        // compute the distance function
        signed_distance_2d(d.as_slice_mut().unwrap(), u.as_slice().unwrap(), dim, h);
        u.clone_from(&d);
    }

    let verts = isosurface::marching_triangles(u.as_slice().unwrap(), dim, 0.);

    for _ in 0..k1 {
        // compute the distance function
        signed_distance_2d(d.as_slice_mut().unwrap(), u.as_slice().unwrap(), dim, h);
        u.clone_from(&d);
    }

    let ex_verts = isosurface::marching_triangles(u.as_slice().unwrap(), dim, 0.);

    let mut fg = Figure::new();
    if let Some(f) = args.flag_svg {
        fg.set_terminal("svg size 1280, 1280", &f);
    }

    {
        {
            let mut axes = fg.axes2d();
            axes.set_aspect_ratio(AutoOption::Fix(1.));

            let grid_opt = [PlotOption::LineStyle(DashType::DotDash)];

            for x in 0..n {
                axes.lines(&[x, x], &[0, n], &grid_opt);
                axes.lines(&[0, n], &[x, x], &grid_opt);
                axes.lines(&[0, x], &[n - x, n], &grid_opt);
                axes.lines(&[x, n], &[0, n - x], &grid_opt);
            }


            for ((verts, c), i) in [init_verts, verts, ex_verts]
                .iter()
                .zip(&["blue", "red", "black"])
                .zip(&[0, k, k + k1])
            {
                let mut q = 2;
                for line in verts.components() {
                    axes.lines(
                        line.iter().map(|p| p[0]),
                        line.iter().map(|p| p[1]),
                        &[PlotOption::Color(c), PlotOption::Caption(&format!("{}", i))][..q],
                    );
                    // add label only for the first line segment
                    if q > 1 {
                        q -= 1;
                    }
                }
            }
        }
        fg.show();
    }
}
