#[macro_use]
extern crate criterion;
extern crate fast_sweeping;

use criterion::{Bencher, Criterion};
use fast_sweeping::level_set;
use fast_sweeping::norm::EuclideanNorm;
use fast_sweeping::*;
use std::time::Duration;

fn bench_2d(b: &mut Bencher, dim: (usize, usize)) {
    let (nx, ny) = dim;
    let mut u = vec![0.; nx * ny];
    let mut d = vec![0.; nx * ny];

    let r = 0.3;
    let hx = 1. / (nx - 1) as f64;
    let hy = 1. / (ny - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * hx - 0.5;
            let y = j as f64 * hy - 0.5;
            u[i * ny + j] = (x * x + y * y).sqrt() - r;
        }
    }

    b.iter(|| {
        signed_distance_2d(&mut d, &u, dim, hx);
    });
}

fn bench_init_2d(b: &mut Bencher, dim: (usize, usize)) {
    let (nx, ny) = dim;
    let mut u = vec![0.; nx * ny];
    let mut d = vec![0.; nx * ny];

    let r = 0.3;
    let hx = 1. / (nx - 1) as f64;
    let hy = 1. / (ny - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * hx - 0.5;
            let y = j as f64 * hy - 0.5;
            u[i * ny + j] = (x * x + y * y).sqrt() - r;
        }
    }

    b.iter(|| {
        level_set::init_dist_2d(&mut d, &u, dim, |p| EuclideanNorm.dual_norm(p));
    });
}

fn bench_init_3d(b: &mut Bencher, dim: (usize, usize, usize)) {
    let (nx, ny, nz) = dim;
    let mut u = vec![0.; nx * ny * nz];
    let mut d = vec![0.; nx * ny * nz];

    let r = 0.3;
    let h = 1. / (nx - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 * h - 0.5;
                let y = j as f64 * h - 0.5;
                let z = k as f64 * h - 0.5;
                u[i * ny * nz + j * nz + k] = (x * x + y * y + z * z).sqrt() - r;
            }
        }
    }

    b.iter(|| {
        level_set::init_dist_3d(&mut d, &u, dim, |p| EuclideanNorm.dual_norm(p));
    });
}

fn bench_3d(b: &mut Bencher, dim: (usize, usize, usize)) {
    let (nx, ny, nz) = dim;
    let mut u = vec![0.; nx * ny * nz];
    let mut d = vec![0.; nx * ny * nz];

    let r = 0.3;
    let h = 1. / (nx - 1) as f64;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 * h - 0.5;
                let y = j as f64 * h - 0.5;
                let z = k as f64 * h - 0.5;
                u[i * ny * nz + j * nz + k] = (x * x + y * y + z * z).sqrt() - r;
            }
        }
    }

    b.iter(|| {
        signed_distance_3d(&mut d, &u, dim, h);
    });
}

fn bench_signed_distance_2d(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "signed_distance_2d",
        |b, &&size| bench_2d(b, (size, size)),
        &[128, 512],
    );
}

fn bench_init_dist_2d(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "init_dist_2d",
        |b, &&size| bench_init_2d(b, (size, size)),
        &[128, 512],
    );
}

fn bench_signed_distance_3d(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "signed_distance_3d",
        |b, &&size| bench_3d(b, (size, size, size)),
        &[32, 64],
    );
}

fn bench_init_dist_3d(c: &mut Criterion) {
    c.bench_function_over_inputs(
        "init_dist_3d",
        |b, &&size| bench_init_3d(b, (size, size, size)),
        &[32, 64],
    );
}

criterion_group!{
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(200))
                .measurement_time(Duration::from_secs(1))
                .sample_size(5);
    targets = bench_signed_distance_2d, bench_init_dist_2d,
                bench_signed_distance_3d, bench_init_dist_3d
}
criterion_main!(benches);
