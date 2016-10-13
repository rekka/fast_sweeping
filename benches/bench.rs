#![feature(test)]
extern crate test;
extern crate fast_sweeping;

mod bench {
    use test::{black_box, Bencher};
    use fast_sweeping::*;

    fn bench_2d(b: &mut Bencher, dim: (usize, usize)) {
        let (nx, ny) = dim;
        let mut u = vec![0.; nx * ny];
        let mut d = black_box(vec![0.; nx * ny]);

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

    #[bench]
    fn s128(b: &mut Bencher) {
        bench_2d(b, (128, 128));
    }

    #[bench]
    fn s512(b: &mut Bencher) {
        bench_2d(b, (512, 512));
    }

    // too slow now
    // #[bench]
    // fn s2048(b: &mut Bencher) {
    //     bench_2d(b, (2048, 2048));
    // }
}
