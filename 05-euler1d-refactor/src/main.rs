/**
 * @brief      Rust implentation of a 2nd-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 * @note       Factors generalizable code used in the previous examples into the
 *             lib_hydro_algorithms crate. The SolutionState struct is now
 *             created by specializing the SolutionStateArray1 generic struct,
 *             which comes already enabled for Runge-Kutta integration via the
 *             WeightedAverage trait. We have also created a generic PLM
 *             function in lib_hydro_algorithms::piecewise_linear, through use
 *             of the IntoAndFromF64Array3 trait, which is implemented for the
 *             lib_euler1d::Primitive struct.
 *
 *             The remaining code demonstates illustrates common boilerplate
 *             functions you expect to see at the top level of a science
 *             program:
 *
 *             1. The main loop
 *             2. The initial state function
 *             3. The update function
 */




// ============================================================================
use ndarray::prelude::*;
use ndarray::{Zip, Axis, stack};
use num::rational::Rational64;
use num::ToPrimitive;

use lib_euler1d::*;
use lib_hydro_algorithms::runge_kutta as rk;
use lib_hydro_algorithms::piecewise_linear::plm_gradient3;
use lib_hydro_algorithms::solution_states::SolutionStateArray1;




// ============================================================================
type SolutionState = SolutionStateArray1<Conserved>;




// ============================================================================
fn write_ascii(state: &SolutionState, filename: String, gamma_law_index: f64, cell_centers: Array1<f64>) {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::LineWriter;
    let file = File::create(filename);
    let mut writer = LineWriter::new(file.unwrap());

    for (x, p) in cell_centers.iter().zip(state.conserved.iter().map(|u| u.to_primitive(gamma_law_index))) {
        writeln!(writer, "{} {} {} {}", x, p.0, p.1, p.2).expect("Write failed");
    }
}




// ============================================================================
fn extend(primitive: Array1<Primitive>) -> Array1<Primitive> {
    let n = primitive.len_of(Axis(0));
    let pl = primitive[0];
    let pr = primitive[n-1];
    stack![Axis(0), [pl, pl], primitive, [pr, pr]]
}




// ============================================================================
fn update(state: SolutionState, gamma_law_index: f64) -> SolutionState {
    let n = state.conserved.len_of(Axis(0));
    let dx = 1.0 / (n as f64);
    let dt = 0.1 * dx;

    let pe = extend(state.conserved.mapv(|u| u.to_primitive(gamma_law_index)));
    let pl = pe.slice(s![ ..-2]);
    let p0 = pe.slice(s![1..-1]);
    let pr = pe.slice(s![2..  ]);
    let dp = azip![pl, p0, pr].apply_collect(|pl, p0, pr| plm_gradient3(2.0, pl, p0, pr)) * 0.5;
    let pfl = &pe.slice(s![1..-2]) + &dp.slice(s![..-1]);
    let pfr = &pe.slice(s![2..-1]) - &dp.slice(s![ 1..]);
    let godunov_fluxes = Zip::from(&pfl).and(&pfr).apply_collect(|&pl, &pr| riemann_hlle(pl, pr, gamma_law_index));

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
fn cell_centers(num_zones: usize) -> Array1<f64> {
    let vertices = Array::<f64, _>::linspace(0.0, 1.0, num_zones + 1);
    let cell_centers = 0.5 * (&vertices.slice(s![1..]) + &vertices.slice(s![..-1]));
    cell_centers
}




// ============================================================================
fn initial_state(num_zones: usize, gamma_law_index: f64) -> SolutionState {
    let initial_cons = cell_centers(num_zones)
        .mapv(|x| if x < 0.5 { Primitive(1.0, 0.0, 1.0) } else { Primitive(0.1, 0.0, 0.125) })
        .mapv(|p| p.to_conserved(gamma_law_index));

    SolutionState {
        time: 0.0,
        iteration: Rational64::new(0, 1),
        conserved: initial_cons,
    }
}




// ============================================================================
fn main() {
    use std::time::Instant;

    let gamma_law_index = 5.0 / 3.0;
    let num_zones = 5000;
    let start_program = Instant::now();

    let mut state = initial_state(num_zones, gamma_law_index);

    while state.time < 0.2 {
        let start = Instant::now();
        state = rk::advance(state, |s| update(s, gamma_law_index), rk::RungeKuttaOrder::RK3);
        println!("[{:05}] t={:.3} kzps={:.3}", state.iteration, state.time, (num_zones as f64) * 1e-3 / start.elapsed().as_secs_f64());
    }

    println!("mean kzps = {:.3}", (num_zones as f64) * 1e-3 * state.iteration.to_f64().unwrap() / start_program.elapsed().as_secs_f64());
    write_ascii(&state, "output.dat".to_string(), gamma_law_index, cell_centers(num_zones));
}
