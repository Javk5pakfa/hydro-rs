/**
 * @brief      Rust implementation of a 2nd-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 * @note       Demonstrates how to use the command line interface to configure
 *             the program's runtime behavior.
 */




// ============================================================================
use ndarray::prelude::*;
use ndarray::{Zip, Axis, stack};
use num::rational::Rational64;
use num::ToPrimitive;
use clap::Clap;

use lib_euler1d::*;
use lib_hydro_algorithms::runge_kutta as rk;
use lib_hydro_algorithms::piecewise_linear::plm_gradient3;
use lib_hydro_algorithms::solution_states::SolutionStateArray1;




// ============================================================================
type SolutionState = SolutionStateArray1<Conserved>;




// ============================================================================
fn write_hdf5(state: &SolutionState, filename: String, gamma_law_index: f64, cell_centers: Array1<f64>) -> Result<(), hdf5::Error> {
    use hdf5::types::VarLenAscii;
    use hdf5::File;

    let file = File::create(filename)?;
    let data = state.conserved.mapv(|u| Into::<[f64; 3]>::into(u.to_primitive(gamma_law_index)));
    file.new_dataset::<[f64; 3]>().create("primitive", data.len_of(Axis(0)))?.write(&data)?;
    file.new_dataset::<f64>().create("cell_centers", cell_centers.len_of(Axis(0)))?.write(&cell_centers)?;
    file.new_dataset::<VarLenAscii>().create("format", ())?.write_scalar(&VarLenAscii::from_ascii("euler1d").unwrap())?;

    Ok(())
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




/**
 * @brief      This data structure describes the runtime configuration of your
 *             program. Clap automatically generates a command-line parser that
 *             will will in the argument values based on inputs on the command
 *             line.
 */
#[derive(Clap)]
#[clap(
    version="0.0.1",
    author="J. Zrake <jzrake@clemson.edu>",
    about="Demonstrates how to configure your program with a command-line interface")]


// ============================================================================
struct Opts {
    #[clap(short, long, default_value="1000", about="The number of grid cells to use")]
    num_zones: usize,

    #[clap(short, long, default_value="0.0", about="The time at which to stop the simulation")]
    tfinal: f64,

    #[clap(short, long, default_value="2", about="The time at which to stop the simulation")]
    rk_order: i64,
}




// ============================================================================
fn main() {
    use std::time::Instant;

    let opts = Opts::parse();
    let gamma_law_index = 5.0 / 3.0;
    let rk_order = match opts.rk_order {
        1 => rk::RungeKuttaOrder::RK1,
        2 => rk::RungeKuttaOrder::RK2,
        3 => rk::RungeKuttaOrder::RK3,
        _ => panic!("Runge-Kutta order must be 1, 2, or 3"),
    };
    let mut state = initial_state(opts.num_zones, gamma_law_index);
    let start_program = Instant::now();

    while state.time < opts.tfinal {
        let start = Instant::now();
        state = rk::advance(state, |s| update(s, gamma_law_index), rk_order);
        println!("[{:05}] t={:.3} kzps={:.3}", state.iteration, state.time, (opts.num_zones as f64) * 1e-3 / start.elapsed().as_secs_f64());
    }

    println!("mean kzps = {:.3}", (opts.num_zones as f64) * 1e-3 * state.iteration.to_f64().unwrap() / start_program.elapsed().as_secs_f64());
    write_hdf5(&state, "output.h5".to_string(), gamma_law_index, cell_centers(opts.num_zones)).expect("HDF5 write failed");
}
