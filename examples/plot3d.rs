extern crate surfviz;
#[macro_use]
extern crate glium;
extern crate isosurface;
extern crate ndarray;

use ndarray::Array;
use isosurface::marching_tetrahedra;
use glium::glutin;

use surfviz::surface;
use surfviz::camera;

fn main() {
    let res = 10;
    let xs = Array::linspace(-0.5, 0.5, res);
    let ys = Array::linspace(-0.5, 0.5, res);
    let zs = Array::linspace(-0.5, 0.5, res);

    let dim = (xs.len(), ys.len(), zs.len());

    let u = {
        let mut u = Array::from_elem(dim, 0.);

        let r = 0.3;
        for ((i, j, k), u) in u.indexed_iter_mut() {
            let (x, y, z) = (xs[i], ys[j], zs[k]);
            *u = x * x + y * y + z * z - r * r;
        }
        u
    };

    let level = 0.0;
    let (verts, faces, normals) = marching_tetrahedra(u.as_slice().unwrap(), dim, level);

    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
        .with_depth_buffer(24)
        .build_glium()
        .unwrap();

    let surface = surface::Surface::new(&display, &verts, &faces, &normals);

    let mut camera = camera::CameraState::new();
    let mut wireframe = false;
    let mut show_normals = false;

    loop {
        camera.update();

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        surface.draw(&mut target, &camera, wireframe, show_normals).unwrap();

        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                             _,
                                             Some(glutin::VirtualKeyCode::F)) => {
                    wireframe = !wireframe
                }
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                             _,
                                             Some(glutin::VirtualKeyCode::N)) => {
                    show_normals = !show_normals
                }
                ev => camera.process_input(&ev),
            }
        }
    }
}
