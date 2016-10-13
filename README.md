# fast_sweeping

The fast sweeping method for the computation of the signed distance function in 2D in 3D.

Based on [1].

### Usage

Add the following to your `Cargo.toml`

```toml
[dependencies.fast_sweeping]
git = "https://github.com/rekka/fast_sweeping.git"
```

At the top of your crate add:

```rust
extern crate fast_sweeping;
```

Depending on the dimension, use `signed_distance_2d` or `signed_distance_3d`.

### Accuracy

There are two main things to consider when evaluating the accuracy of the method.

#### Initialization near the level set

There is no unique way of doing this. In this implementation, we simply split the squares/cubes
of the regular mesh into 2 triangles/6 tetrahedra and assume that the level set function is
linear on each of them. The initial distance at the vertices is then taken as the distance to
the line/plane going through the triangle/tetrahedron (_not_ to the intersection of the
line/plane with the triangle/tetrahedron). If the vertex is contained in multiple
triangles/tetrahedra that intersect the level set, the minimum of all distances is taken.

The main advantages of this approach are:

- Simple.
- Does _not_ move _flat_ parts of the level set (unless near other parts of the level set).
- Does not move symmetric corners.

Disadvantages:

- Introduces some more anisotropy.
- The initial value outside of corners is not really the distance to the level set but smaller.
But this does not seem to actually cause larger error in the circle test case, see
`examples/error`. The error seems to be bigger _inside_ a circle. It appears that within small
neighborhood (dist <= 3h) of the level set the max error of order h².

For an example see `examples/redistance`.

#### Finite difference approximation

Here we use the simplest first order upwind numerical discretization as given in [1]. This
gives an error of order `O(|h log h|)`.

### Performance

The performance is limited by the speed of computing `sqrt` during the sweeps. Possible
future optimizations are:

  - Only compute distance in a small neighborhood of the level set.
  - Compute two square roots in one instruction `sqrtpd`.
  - Use multiple threads. However, this is relatively nontrivial due to the sequential nature of
    the Gauss-Seidel iteration.

### References

[1] Zhao, Hongkai A fast sweeping method for eikonal equations. Math. Comp. 74 (2005), no. 250,
603–627.

## License

MIT license: see the `LICENSE` file for details.
