/**
 * @brief      Rust implentation of a 1st-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 * @note       This version demonstrates use of the local lib-euler1d crate.
 *             Note that performance is ~50% slower than the no-import version,
 *             unless the build profile enables link-time-optimizations "lto".
 */




// ============================================================================
use lib_euler1d::*;
use ndarray::prelude::*;
use ndarray::{Zip, Axis, stack};




// ============================================================================
struct SolutionState {
    time: f64,
    iteration: i64,
    conserved: Array1<Conserved>,
}

impl SolutionState {
    fn write_ascii(self, filename: String, gamma_law_index: f64, cell_centers: Array1<f64>) {
        use std::fs::File;
        use std::io::prelude::*;
        use std::io::LineWriter;
        let file = File::create(filename);
        let mut writer = LineWriter::new(file.unwrap());

        for (x, p) in cell_centers.iter().zip(self.conserved.iter().map(|u| u.to_primitive(gamma_law_index))) {
            writeln!(writer, "{} {} {} {}", x, p.0, p.1, p.2).expect("Write failed");
        }
    }
}




// ============================================================================
fn extend(primitive: Array1<Primitive>) -> Array1<Primitive> {
    stack(Axis(0), &[primitive.slice(s![..1]), primitive.view(), primitive.slice(s![-1..])]).unwrap()
}

fn update(state: SolutionState, gamma_law_index: f64) -> SolutionState {
    let n = state.conserved.len_of(Axis(0));
    let dx = 1.0 / (n as f64);
    let dt = 0.1 * dx;

    let primitive = extend(state.conserved.mapv(|u| u.to_primitive(gamma_law_index)));
    let pl = primitive.slice(s![..-1]);
    let pr = primitive.slice(s![ 1..]);
    let godunov_fluxes = Zip::from(pl).and(pr).apply_collect(|&pl, &pr| riemann_hlle(pl, pr, gamma_law_index));

    let gl = &godunov_fluxes.slice(s![..-1]);
    let gr = &godunov_fluxes.slice(s![ 1..]);
    let du = (gl - gr) * (dt / dx);

    SolutionState {
        time:      state.time + dt,
        iteration: state.iteration + 1,
        conserved: state.conserved + du,
    }
}




// ============================================================================
fn main() {
    use std::time::Instant;

    let gamma_law_index = 5.0 / 3.0;
    let num_zones = 5000;
    let vertices = Array::<f64, _>::linspace(0.0, 1.0, num_zones + 1);
    let cell_centers = 0.5 * (&vertices.slice(s![1..]) + &vertices.slice(s![..-1]));
    let initial_cons = cell_centers
        .mapv(|x| if x < 0.5 { Primitive(1.0, 0.0, 1.0) } else { Primitive(0.1, 0.0, 0.125) })
        .mapv(|p| p.to_conserved(gamma_law_index));

    let mut state = SolutionState {
        time: 0.0,
        iteration: 0,
        conserved: initial_cons,
    };

    let start_program = Instant::now();

    while state.time < 0.2 {
        let start = Instant::now();
        state = update(state, gamma_law_index);
        println!("[{:05}] t={:.3} kzps={:.3}", state.iteration, state.time, (num_zones as f64) * 1e-3 / start.elapsed().as_secs_f64());
    }

    println!("mean kzps = {:.3}", (num_zones as f64) * 1e-3 * (state.iteration as f64) / start_program.elapsed().as_secs_f64());
    state.write_ascii("output.dat".to_string(), gamma_law_index, cell_centers);
}
