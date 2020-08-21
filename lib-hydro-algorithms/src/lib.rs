use std::ops::{Add, Mul};
use num::rational::Rational64;

pub mod solution_states;
pub mod piecewise_linear;
pub mod runge_kutta;

pub trait IntoAndFromF64Array3 {
	fn into_f64_array3(self) -> [f64; 3];
	fn from_f64_array3(a: [f64; 3]) -> Self;
}

pub trait WeightedAverage: Clone + Add<Output=Self> + Mul<Rational64, Output=Self>
{
}
