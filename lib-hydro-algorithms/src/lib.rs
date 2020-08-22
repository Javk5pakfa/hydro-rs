use std::ops::{Add, Mul};
use num::rational::Rational64;

pub mod solution_states;
pub mod piecewise_linear;
pub mod runge_kutta;

pub trait WeightedAverage: Clone + Add<Output=Self> + Mul<Rational64, Output=Self> {}
