extern crate fast_sweeping;
extern crate surfviz;
#[macro_use]
extern crate glium;
extern crate isosurface;
extern crate ndarray;
extern crate time;

use ndarray::Array;
use isosurface::marching_tetrahedra;
use glium::glutin;
use time::PreciseTime;

use surfviz::surface;
use surfviz::camera;

fn main() {
    let res = 10;
    let h = 1. / (res - 1)  as f64;
    let xs: Array<f64, _> = Array::linspace(-0.5, 0.5, res);
    let ys = Array::linspace(-0.5, 0.5, res);
    let zs = Array::linspace(-0.5, 0.5, res);

    let dim = (xs.len(), ys.len(), zs.len());

    let u = {
        let mut u: Array<f64, _> = Array::from_elem(dim, 0.);

        let r = 0.3;
        for ((i, j, k), u) in u.indexed_iter_mut() {
            let (x, y, z) = (xs[i], ys[j], zs[k]);
            *u = (x * x + y * y + z * z).sqrt() - r;
            // *u = u.min((x - 0.25) * (x - 0.25) + y * y + z * z - 0.2 * 0.2);
            // *u  = (x + y + z) / (3f64).sqrt();
            // *u  = (x + y) / (2f64).sqrt();
        }
        u
    };

    let mut d = u.clone();

    let mut v = u.clone();
    for _ in 0..1 {
        fast_sweeping::signed_distance_3d(d.as_slice_mut().unwrap(), v.as_slice().unwrap(), dim, h);
        v.clone_from(&d);
    }

    let level = 0.2;

    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new()
        .with_depth_buffer(24)
        .build_glium()
        .unwrap();

    let (verts, faces, normals) = marching_tetrahedra(u.as_slice().unwrap(), dim, level);
    let surface = surface::Surface::new(&display, &verts, &faces, &normals);

    let (verts, faces, normals) = marching_tetrahedra(d.as_slice().unwrap(), dim, level);
    let signed_distance_surface = {
        let mut s = surface::Surface::new(&display, &verts, &faces, &normals);
        s.set_color([0., 0., 1., 1.]);
        let m: [[f32; 4]; 4] = surface.model_matrix();
        s.set_model_matrix(m);
        s
    };

    let mut camera = camera::CameraState::new();
    let mut wireframe = false;
    let mut show_normals = false;

    let mut frame_time = PreciseTime::now();

    loop {
        camera.update();

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        surface.draw(&mut target, &camera, wireframe, show_normals).unwrap();
        signed_distance_surface.draw(&mut target, &camera, wireframe, show_normals).unwrap();

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

        // sleep if fps to big
        let now = PreciseTime::now();
        let t = frame_time.to(now).num_milliseconds();
        let fps = 60;
        let ft = 1000i64 / fps;

        if t < ft {
            std::thread::sleep(std::time::Duration::from_millis((ft - t) as u64));
        }
        frame_time = now;
    }
}
