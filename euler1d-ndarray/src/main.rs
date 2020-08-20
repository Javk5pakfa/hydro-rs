/**
 * @brief      Rust implentation of a 1st-order Godunov code to solve the 1D
 *             Euler equation.
 *
 * @copyright  Jonathan Zrake, Clemson University, 2020
 *
 * @note       This implementation of the 1d hydro code uses the ndarray crate.
 *             It is ~15-20% slower than the nodeps version.
 */




// ============================================================================
#[derive(Copy, Clone)]
struct Conserved(f64, f64, f64);

#[derive(Copy, Clone)]
struct Primitive(f64, f64, f64);




// ============================================================================
impl std::ops::Add<Conserved> for Conserved { type Output = Self; fn add(self, u: Conserved) -> Conserved { Conserved(self.0 + u.0, self.1 + u.1, self.2 + u.2) } }
impl std::ops::Sub<Conserved> for Conserved { type Output = Self; fn sub(self, u: Conserved) -> Conserved { Conserved(self.0 - u.0, self.1 - u.1, self.2 - u.2) } }
impl std::ops::Mul<Conserved> for f64 { type Output = Conserved; fn mul(self, u: Conserved) -> Conserved { Conserved(self * u.0, self * u.1, self * u.2) } }
impl std::ops::Mul<f64> for Conserved { type Output = Conserved; fn mul(self, a: f64) -> Conserved { Conserved(self.0 * a, self.1 * a, self.2 * a) } }
impl std::ops::Div<f64> for Conserved { type Output = Conserved; fn div(self, a: f64) -> Conserved { Conserved(self.0 / a, self.1 / a, self.2 / a) } }




// ============================================================================
impl Conserved {
    fn density        (self)  -> f64 { self.0 }
    fn momentum       (self)  -> f64 { self.1 }
    fn total_energy   (self)  -> f64 { self.2 }
    fn kinetic_energy (self)  -> f64 { self.momentum().powi(2) / (2.0 * self.density()) }
    fn thermal_energy (self)  -> f64 { self.total_energy() - self.kinetic_energy() }
    fn to_primitive   (self, gamma_law_index: f64) -> Primitive {
        Primitive(
            self.density(),
            self.momentum() / self.density(),
            self.thermal_energy() * (gamma_law_index - 1.0))
    }
}




// ============================================================================
impl Primitive {
    fn density (self) -> f64 { self.0 }
    fn velocity(self) -> f64 { self.1 }
    fn pressure(self) -> f64 { self.2 }

    fn momentum(self) -> f64 {
        self.density() * self.velocity()
    }

    fn total_energy(self, gamma_law_index: f64) -> f64 {
        let kinetic = 0.5 * self.density() * self.velocity().powi(2);
        let thermal = self.pressure() / (gamma_law_index - 1.0);
        kinetic + thermal
    }

    fn sound_speed_squared(self, gamma_law_index: f64) -> f64 {
        gamma_law_index * self.pressure() / self.density()
    }

    fn to_conserved(self, gamma_law_index: f64) -> Conserved {
        Conserved(
            self.density(),
            self.momentum(),
            self.total_energy(gamma_law_index))
    }

    fn outer_wavespeeds(self, gamma_law_index: f64) -> (f64, f64) {
        let cs = self.sound_speed_squared(gamma_law_index).sqrt();
        let vn = self.velocity();
        (vn - cs, vn + cs)
    }

    fn flux_vector(self, gamma_law_index: f64) -> Conserved {
        let pg = self.pressure();
        let vn = self.velocity();
        let advective_term = vn * self.to_conserved(gamma_law_index);
        let pressure_term = Conserved(0.0, pg, pg * vn);
        advective_term + pressure_term
    }
}




// ============================================================================
fn riemann_hlle(pl: Primitive, pr: Primitive, gamma_law_index: f64) -> Conserved {
    let ul = pl.to_conserved(gamma_law_index);
    let ur = pr.to_conserved(gamma_law_index);
    let fl = pl.flux_vector(gamma_law_index);
    let fr = pr.flux_vector(gamma_law_index);

    let (alm, alp) = pl.outer_wavespeeds(gamma_law_index);
    let (arm, arp) = pr.outer_wavespeeds(gamma_law_index);
    let ap = alp.max(arp).max(0.0);
    let am = alm.min(arm).min(0.0);

    (ap * fl - am * fr - (ul - ur) * ap * am) / (ap - am)
}




// ============================================================================
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
