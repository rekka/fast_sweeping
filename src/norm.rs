use super::{max, min};

/// Trait for setting up anisotropic distance function computation.
///
/// Represents the __dual norm__ of the desired anisotropic norm.
///
/// `V` is the vector type, either `[S; 2]` or `[S; 3]` and `S` is the scalar type, usually `f64`.
///
pub trait DualNorm<V, S> {
    /// `dual_norm` returns the __dual__ norm. It must be a positively one-homogeneous function.
    fn dual_norm(&self, p: V) -> S;
    /// Solves the "inverse problem" for the norm: Given values `v_i` at points `-s_i e_i` (`e_i`
    /// is the canonical basis), find the largest value `t ≤ d` at the origin such that `||p|| ≤
    /// 1`, where `p_i = s_i (t - v_i)_+` and `||p||` is the __dual__ anisotropic norm.
    fn inv_dual_norm(&self, d: S, v: V, s: V) -> S;
}

/// Euclidean (l^2) norm
pub struct EuclideanNorm;

/// Dual norm for the max (l^∞) norm is the l^1 norm.
impl DualNorm<[f64; 2], f64> for EuclideanNorm {
    fn dual_norm(&self, p: [f64; 2]) -> f64 {
        (p[0] * p[0] + p[1] * p[1]).sqrt()
    }

    fn inv_dual_norm(&self, d: f64, v: [f64; 2], _: [f64; 2]) -> f64 {
        let a = v[0];
        let b = v[1];

        let x = if (a - b).abs() >= 1. {
            min(a, b) + 1.
        } else {
            0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt())
        };

        min(d, x)
    }
}

/// Dual norm for the max (l^∞) norm is the l^1 norm.
impl DualNorm<[f64; 3], f64> for EuclideanNorm {
    fn dual_norm(&self, p: [f64; 3]) -> f64 {
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    fn inv_dual_norm(&self, d: f64, v: [f64; 3], _: [f64; 3]) -> f64 {
        let (a, b, c) = {
            use std::mem::swap;
            let mut a = v[0];
            let mut b = v[1];
            let mut c = v[2];

            if a > b {
                swap(&mut a, &mut b);
            }
            if b > c {
                swap(&mut b, &mut c);
            }
            if a > b {
                swap(&mut a, &mut b);
            }

            (a, b, c)
        };

        let x = if b >= a + 1. {
            a + 1.
        } else {
            let x = 0.5 * (a + b + (2. - (a - b) * (a - b)).sqrt());
            if x <= c {
                x
            } else {
                let v = (1. / 3.)
                    * (a + b + c
                        + (3. + (a + b + c).powi(2) - 3. * (a * a + b * b + c * c)).sqrt());
                v
            }
        };

        min(d, x)
    }
}

/// l^1 (taxicab, Manhattan) norm
pub struct L1Norm;

/// Dual norm for the l^1 norm is the l^∞ norm.
impl DualNorm<[f64; 2], f64> for L1Norm {
    fn dual_norm(&self, p: [f64; 2]) -> f64 {
        max(p[0].abs(), p[1].abs())
    }

    fn inv_dual_norm(&self, d: f64, v: [f64; 2], _: [f64; 2]) -> f64 {
        min(d, min(v[0], v[1]) + 1.)
    }
}

/// Maximum (l^∞) norm
pub struct MaxNorm;

/// Dual norm for the max (l^∞) norm is the l^1 norm.
impl DualNorm<[f64; 2], f64> for MaxNorm {
    fn dual_norm(&self, p: [f64; 2]) -> f64 {
        p[0].abs() + p[1].abs()
    }

    fn inv_dual_norm(&self, d: f64, v: [f64; 2], _: [f64; 2]) -> f64 {
        min(min(d, v[0] + 1.), min(v[1] + 1., 0.5 * (v[0] + v[1] + 1.)))
    }
}

/// Dual norm for the max (l^∞) norm is the l^1 norm.
impl DualNorm<[f64; 3], f64> for MaxNorm {
    fn dual_norm(&self, p: [f64; 3]) -> f64 {
        p[0].abs() + p[1].abs() + p[2].abs()
    }

    fn inv_dual_norm(&self, d: f64, v: [f64; 3], _: [f64; 3]) -> f64 {
        let (a, b, c) = {
            use std::mem::swap;
            let mut a = v[0];
            let mut b = v[1];
            let mut c = v[2];

            if a > b {
                swap(&mut a, &mut b);
            }
            if b > c {
                swap(&mut b, &mut c);
            }
            if a > b {
                swap(&mut a, &mut b);
            }

            (a, b, c)
        };
        min(
            min(d, a + 1.),
            min(0.5 * (a + b + 1.), (1. / 3.) * (a + b + c + 1.)),
        )
    }
}

/// Convenience function to test the consistency of the `inv_dual_norm` implementation with
/// `dual_norm` by generating vectors on a `(n + 1)^3` grid with values in the interval [-m, m].
pub fn test_inv_dual_norm_2d<N>(norm: N, m: f64, n: u32)
where
    N: DualNorm<[f64; 2], f64>,
{
    let h = 2. * m / n as f64;

    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            let v = [i as f64 * h - m, j as f64 * h - m];
            let s = [-v[0].signum(), -v[1].signum()];
            let t = norm.inv_dual_norm(0., v, s);
            assert!(t <= 0., "returned t > d");

            let x = norm.dual_norm([s[0] * (t - v[0]).max(0.), s[1] * (t - v[1]).max(0.)]);
            if t < 0. {
                assert!((x - 1.).abs() < 1e-7, "t = {}, v = {:?}, s = {:?}", t, v, s);
            } else {
                assert!(t < 1. + 1e-7, "t = {}, v = {:?}, s = {:?}", t, v, s);
            }
        }
    }
}

/// Convenience function to test the consistency of the `inv_dual_norm` implementation with
/// `dual_norm` by generating vectors on a `(n + 1)^3` grid with values in the interval [-m, m].
pub fn test_inv_dual_norm_3d<N>(norm: N, m: f64, n: u32)
where
    N: DualNorm<[f64; 3], f64>,
{
    let h = 2. * m / n as f64;

    for i in 0..(n + 1) {
        for j in 0..(n + 1) {
            for k in 0..(n + 1) {
                let v = [i as f64 * h - m, j as f64 * h - m, k as f64 * h - m];
                let s = [-v[0].signum(), -v[1].signum(), -v[2].signum()];
                let t = norm.inv_dual_norm(0., v, s);
                assert!(t <= 0.);

                let x = norm.dual_norm([
                    s[0] * (t - v[0]).max(0.),
                    s[1] * (t - v[1]).max(0.),
                    s[2] * (t - v[2]).max(0.),
                ]);
                if t < 0. {
                    assert!(
                        (x - 1.).abs() < 1e-7,
                        "dual_norm = {}, t = {}, v = {:?}, s = {:?}",
                        x,
                        t,
                        v,
                        s
                    );
                } else {
                    assert!(
                        t < 1. + 1e-7,
                        "dual_norm = {}, t = {}, v = {:?}, s = {:?}",
                        x,
                        t,
                        v,
                        s
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_norm_max_norm() {
        test_inv_dual_norm_2d(MaxNorm, 2., 5);
        test_inv_dual_norm_3d(MaxNorm, 2., 5);
    }

    #[test]
    fn dual_norm_euclidean_norm() {
        test_inv_dual_norm_2d(EuclideanNorm, 2., 5);
        test_inv_dual_norm_3d(EuclideanNorm, 2., 5);
    }

    #[test]
    fn dual_norm_l1_norm() {
        test_inv_dual_norm_2d(L1Norm, 2., 5);
    }
}
