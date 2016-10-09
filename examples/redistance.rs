//! Illustrates the effect of an iterative application of the signed distance computation to a
//! curve.
extern crate fast_sweeping;
extern crate ndarray;
extern crate gnuplot;
extern crate isosurface;

use ndarray::prelude::*;
use ndarray::Data;
use fast_sweeping::signed_distance_2d;
#[allow(unused_imports)]
use gnuplot::{Figure, Caption, Color, Fix, AxesCommon, PlotOption, DashType, Coordinate,
              TextColor, ContourStyle, AutoOption};

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

    let k = 10;
    let k1 = 100;

    let r = 0.3;

    let xs: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let ys: Array<f64, _> = Array::linspace(-0.5, 0.5, n + 1);
    let mut u = tensor_product(&xs, &ys, |x, y| (x * x + y * y).sqrt() - r + 0.2 * (x * 12.).sin());

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


            for ((verts, c), i) in [init_verts, verts, ex_verts].iter().zip(&["blue", "red", "black"]).zip(&[0, k, k + k1]) {
                let mut q = 2;
                for line in verts.chunks(2) {
                    axes.lines(line.iter().map(|p| p[0]), line.iter().map(|p| p[1]),
                        &[PlotOption::Color(c),
                            PlotOption::Caption(&format!("{}", i))
                        ][..q]);
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
