use core::ops::{Add, Mul};
use ndarray::Array1;
use num::rational::Rational64;
use num::ToPrimitive;




//============================================================================
#[derive(Clone)]
pub struct SolutionStateArray1<T> {
    pub time: f64,
    pub iteration: Rational64,
    pub conserved: Array1<T>,
}

impl<T> Add for SolutionStateArray1<T> where Array1<T>: Add<Array1<T>, Output=Array1<T>> {
    type Output = SolutionStateArray1<T>;
    fn add(self, b: Self::Output) -> Self::Output {
        Self::Output{
            time: self.time + b.time,
            iteration: self.iteration + b.iteration,
            conserved: self.conserved + b.conserved,
        }
    }
}

impl<T> Mul<Rational64> for SolutionStateArray1<T> where Array1<T>: Mul<f64, Output=Array1<T>> {
    type Output = SolutionStateArray1<T>;
    fn mul(self, b: Rational64) -> Self::Output {
        Self::Output{
            time: self.time * b.to_f64().unwrap(),
            iteration: self.iteration * b,
            conserved: self.conserved * b.to_f64().unwrap(),
        }
    }
}
