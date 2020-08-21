use num::rational::Rational64;
use crate::WeightedAverage;




//============================================================================
pub enum RungeKuttaOrder {
    RK1,
    RK2,
    RK3,
}

pub fn advance_rk1<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    update(s0)
}

pub fn advance_rk2<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    let b0 = Rational64::new(0, 1);
    let b1 = Rational64::new(1, 2);
    let mut s1 = s0.clone();
    s1 = s0.clone() * b0 + update(s1) * (-b0 + 1);
    s1 = s0.clone() * b1 + update(s1) * (-b1 + 1);
    s1
}

pub fn advance_rk3<State, Update>(s0: State, update: Update) -> State where State: WeightedAverage, Update: Fn(State) -> State
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

pub fn advance<State, Update>(state: State, update: Update, order: RungeKuttaOrder) -> State where State: WeightedAverage, Update: Fn(State) -> State
{
    match order {
        RungeKuttaOrder::RK1 => advance_rk1(state, update),        
        RungeKuttaOrder::RK2 => advance_rk2(state, update),        
        RungeKuttaOrder::RK3 => advance_rk3(state, update),        
    }
}
