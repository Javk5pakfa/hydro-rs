/**
 * @brief      Rust implementation of a 2nd-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 * @note       Demonstrates an alternative to using Clap as your runtime
 *             configuration engine. The alternative is a minimal, home-spun
 *             crate called lib_config. The key difference with Clap is that
 *             only key=value pairs are accepted on the command line, no --flags
 *             are allowed. This can be an advantage when you are trying to
 *             configure a physical model with many parameters; the two dashes
 *             in the --parameter=value syntax can become tiring to look at.
 *
 *             The lib_config approach plays nicely with serialization to and
 *             from HDF5, so it's a useful construct when your program supports
 *             restart files (i.e. checkpoints), which we'll get to in a later
 *             tutorial.
 *
 *             One downside is that your runtime configuration is stored as
 *             HashMap of lib_config::Value objects, which at runtime can be
 *             either i64, f64, or String (they are enum's). If you have a Value
 *             x = Value::from(12), its runtime kind will be i64, and calling
 *             e.g. x.to_bool() is a bug and would result in a panic. You are
 *             perfectly welcome to load your configuration into a corresponding
 *             data structure to get stricter typing: the cost would simply be
 *             keeping members of the data structure and the code which defines
 *             the lib_config::Form in sync. Of course you could use Rusts'
 *             macro system to do that automatically, but that's an advanced
 *             topic and I don't know how yet.
 *
 *             You might also consider combining Clap and lib_config, because
 *             there is a natural separation of concerns there: options that
 *             control the program execution, such as parallelization strategy,
 *             verbosity, printing help messages, dry runs, etc --- anything
 *             that does not influence the numerical result --- are flags and
 *             should be provided through Clap. Meanwhile key=value pairs are
 *             model parameters, which do influence the numerical output, and
 *             must be reproducible for science reasons (and thus need to be
 *             stored in the HDF5 checkpoint files). Model parameters should be
 *             routed to lib_config. You can configure Clap to allow any number
 *             of positional (not --flag) arguments, which you can forward to
 *             lib_config as model parameters. An exmaple invocation would look
 *             like
 *             
 *             ./my_program --quiet --threads=32 viscosity=1e-3 domain_radius=24
 */




// ============================================================================
use ndarray::prelude::*;
use ndarray::{Array1, Zip, Axis, stack};
use num::rational::Rational64;
use num::ToPrimitive;

use lib_config;
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




// ============================================================================
fn make_opts() -> Result<lib_config::Form, lib_config::ConfigError> {

    /*
     * The call below transforms a sequence of strings, each having the form
     * "key=val", into a HashMap::<String, String>. The result is an error if
     * any of the string don't have exactly one equals sign, or if redundant
     * keys are encountered. In this example we invoke the function with the
     * program's command line arguments, but these may also be loaded from a
     * file.
     */
    let arg_key_vals = lib_config::to_string_map_from_key_val_pairs(std::env::args().skip(1))?;

    /*
     * All items must have a default value. The final string is an about message
     * which you can print back to the user. One or more hash maps may be merged
     * in after the form is populated. The result of the merge is an error if
     * the hash map contains any keys that were not declared previously by
     * calling 'item'.
     */
    lib_config::Form::new()
        .item("num_zones" , 5000   , "Number of grid cells to use")
        .item("tfinal"    , 0.2    , "Time at which to stop the simulation")
        .item("rk_order"  , 2      , "Runge-Kutta time integration order")
        .item("quiet"     , false  , "Suppress the iteration message")
        .merge_string_map(arg_key_vals)
}




// ============================================================================
fn run() -> Result<(), lib_config::ConfigError> {
    use std::time::Instant;
    let opts = make_opts()?;

    /*
     * Print the configured runtime options to the terminal, along with the
     * about message.
     */
    for (key, parameter) in &opts {
        println!("\t{:.<24} {: <8} {}", key, parameter.value, parameter.about);
    }

    let gamma_law_index = 5.0 / 3.0;
    let num_zones = opts.get("num_zones").as_int() as usize;
    let rk_order = match opts.get("rk_order").as_int() {
        1 => rk::RungeKuttaOrder::RK1,
        2 => rk::RungeKuttaOrder::RK2,
        3 => rk::RungeKuttaOrder::RK3,
        _ => panic!("Runge-Kutta order must be 1, 2, or 3"),
    };
    let mut state = initial_state(num_zones, gamma_law_index);
    let start_program = Instant::now();

    while state.time < opts.get("tfinal").as_float() {
        let start = Instant::now();
        state = rk::advance(state, |s| update(s, gamma_law_index), rk_order);

        if ! opts.get("quiet").as_bool() {
            println!("[{:05}] t={:.3} kzps={:.3}", state.iteration, state.time, (num_zones as f64) * 1e-3 / start.elapsed().as_secs_f64());
        }
    }

    println!("mean kzps = {:.3}", (num_zones as f64) * 1e-3 * state.iteration.to_f64().unwrap() / start_program.elapsed().as_secs_f64());
    write_hdf5(&state, "output.h5".to_string(), gamma_law_index, cell_centers(num_zones)).expect("HDF5 write failed");

    Ok(())
}




// ============================================================================
fn main() {
    run().unwrap_or_else(|error| println!("{}", error));
}
