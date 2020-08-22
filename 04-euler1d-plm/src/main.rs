/**
 * @brief      Rust implementation of a 2nd-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 * @note       Demonstrates a 1D, second-order PLM-based hydro code. It also
 *             introduces explicit RK time-stepping through a generic advance_rk
 *             function. The solution state type must satisfy the custom trait
 *             WeightedAverage. In this program we also introduce the Rational64
 *             type from num::rational, which is used to track fractional
 *             iterations you get with Runge-Kutta time stepping.
 */




// ============================================================================
use ndarray::prelude::*;
use ndarray::{Zip, Axis, stack};
use num::rational::Rational64;
use num::ToPrimitive;
use std::ops::{Add, Mul};
use lib_euler1d::*;




// ============================================================================
pub enum RungeKuttaOrder {
    RK1,
    RK2,
    RK3,
}

trait WeightedAverage: Clone + Add<Output=Self> + Mul<Rational64, Output=Self>
{
}

fn advance_rk1<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    update(s0)
}

fn advance_rk2<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    let b0 = Rational64::new(0, 1);
    let b1 = Rational64::new(1, 2);
    let mut s1 = s0.clone();
    s1 = s0.clone() * b0 + update(s1) * (-b0 + 1);
    s1 = s0.clone() * b1 + update(s1) * (-b1 + 1);
    s1
}

fn advance_rk3<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    let b0 = Rational64::new(0, 1);
    let b1 = Rational64::new(3, 4);
    let b2 = Rational64::new(1, 3);
    let mut s1 = s0.clone();
    s1 = s0.clone() * b0 + update(s1) * (-b0 + 1);
    s1 = s0.clone() * b1 + update(s1) * (-b1 + 1);
    s1 = s0.clone() * b2 + update(s1) * (-b2 + 1);
    s1
}

fn advance_rk<State, Update>(state: State, update: Update, order: RungeKuttaOrder) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    match order {
        RungeKuttaOrder::RK1 => advance_rk1(state, update),        
        RungeKuttaOrder::RK2 => advance_rk2(state, update),        
        RungeKuttaOrder::RK3 => advance_rk3(state, update),        
    }
}




// ============================================================================
#[derive(Clone)]
struct SolutionState {
    time: f64,
    iteration: Rational64,
    conserved: Array1<Conserved>,
}

impl Add for SolutionState {
    type Output = SolutionState;
    fn add(self, b: SolutionState) -> SolutionState {
        SolutionState{
            time: self.time + b.time,
            iteration: self.iteration + b.iteration,
            conserved: self.conserved + b.conserved,
        }
    }
}

impl Mul<Rational64> for SolutionState {
    type Output = SolutionState;
    fn mul(self, b: Rational64) -> SolutionState {
        SolutionState {
            time: self.time * b.to_f64().unwrap(),
            iteration: self.iteration * b,
            conserved: self.conserved * b.to_f64().unwrap(),
        }
    }
}

impl WeightedAverage for SolutionState {
}




// ============================================================================
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
fn plm_gradient(theta: f64, yl: &Primitive, y0: &Primitive, yr: &Primitive) -> Primitive {

    fn sgn(a: f64) -> f64 {
        1.0f64.copysign(a)
    }

    fn minabs(a: f64, b: f64, c: f64) -> f64 {
        a.abs().min(b.abs()).min(c.abs())
    }

    fn plm_gradient_f64(theta: f64, yl: f64, y0: f64, yr: f64) -> f64 {
        let a = (y0 - yl) * theta;
        let b = (yr - yl) * 0.5;
        let c = (yr - y0) * theta;
        0.25 * (sgn(a) + sgn(b)).abs() * (sgn(a) + sgn(c)) * minabs(a, b, c)
    }

    let p0 = plm_gradient_f64(theta, yl.0, y0.0, yr.0);
    let p1 = plm_gradient_f64(theta, yl.1, y0.1, yr.1);
    let p2 = plm_gradient_f64(theta, yl.2, y0.2, yr.2);

    Primitive(p0, p1, p2)
}




// ============================================================================
fn extend(primitive: Array1<Primitive>) -> Array1<Primitive> {
    let n = primitive.len_of(Axis(0));
    let pl = primitive[0];
    let pr = primitive[n-1];
    stack![Axis(0), [pl, pl], primitive, [pr, pr]]
}

fn update(state: SolutionState, gamma_law_index: f64) -> SolutionState {
    let n = state.conserved.len_of(Axis(0));
    let dx = 1.0 / (n as f64);
    let dt = 0.1 * dx;

    let pe = extend(state.conserved.mapv(|u| u.to_primitive(gamma_law_index)));
    let pl = pe.slice(s![ ..-2]);
    let p0 = pe.slice(s![1..-1]);
    let pr = pe.slice(s![2..  ]);
    let dp = azip![pl, p0, pr].apply_collect(|pl, p0, pr| plm_gradient(2.0, pl, p0, pr)) * 0.5;
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
        iteration: Rational64::new(0, 1),
        conserved: initial_cons,
    };

    let start_program = Instant::now();

    while state.time < 0.2 {
        let start = Instant::now();
        state = advance_rk(state, |s| update(s, gamma_law_index), RungeKuttaOrder::RK2);
        println!("[{:05}] t={:.3} kzps={:.3}", state.iteration, state.time, (num_zones as f64) * 1e-3 / start.elapsed().as_secs_f64());
    }

    println!("mean kzps = {:.3}", (num_zones as f64) * 1e-3 * state.iteration.to_f64().unwrap() / start_program.elapsed().as_secs_f64());
    state.write_ascii("output.dat".to_string(), gamma_law_index, cell_centers);
}
