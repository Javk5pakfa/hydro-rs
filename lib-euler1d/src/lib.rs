// ============================================================================
#[derive(Copy, Clone)]
pub struct Conserved(pub f64, pub f64, pub f64);

#[derive(Copy, Clone)]
pub struct Primitive(pub f64, pub f64, pub f64);




// ============================================================================
impl std::ops::Add<Primitive> for Primitive { type Output = Self; fn add(self, u: Primitive) -> Primitive { Primitive(self.0 + u.0, self.1 + u.1, self.2 + u.2) } }
impl std::ops::Sub<Primitive> for Primitive { type Output = Self; fn sub(self, u: Primitive) -> Primitive { Primitive(self.0 - u.0, self.1 - u.1, self.2 - u.2) } }
impl std::ops::Mul<Primitive> for f64 { type Output = Primitive; fn mul(self, u: Primitive) -> Primitive { Primitive(self * u.0, self * u.1, self * u.2) } }
impl std::ops::Mul<f64> for Primitive { type Output = Primitive; fn mul(self, a: f64) -> Primitive { Primitive(self.0 * a, self.1 * a, self.2 * a) } }
impl std::ops::Div<f64> for Primitive { type Output = Primitive; fn div(self, a: f64) -> Primitive { Primitive(self.0 / a, self.1 / a, self.2 / a) } }




// ============================================================================
impl std::ops::Add<Conserved> for Conserved { type Output = Self; fn add(self, u: Conserved) -> Conserved { Conserved(self.0 + u.0, self.1 + u.1, self.2 + u.2) } }
impl std::ops::Sub<Conserved> for Conserved { type Output = Self; fn sub(self, u: Conserved) -> Conserved { Conserved(self.0 - u.0, self.1 - u.1, self.2 - u.2) } }
impl std::ops::Mul<Conserved> for f64 { type Output = Conserved; fn mul(self, u: Conserved) -> Conserved { Conserved(self * u.0, self * u.1, self * u.2) } }
impl std::ops::Mul<f64> for Conserved { type Output = Conserved; fn mul(self, a: f64) -> Conserved { Conserved(self.0 * a, self.1 * a, self.2 * a) } }
impl std::ops::Div<f64> for Conserved { type Output = Conserved; fn div(self, a: f64) -> Conserved { Conserved(self.0 / a, self.1 / a, self.2 / a) } }




// ============================================================================
impl Into<[f64; 3]> for Primitive {
    fn into(self) -> [f64; 3] {
        [self.0, self.1, self.2]
    }    
}

impl From<[f64; 3]> for Primitive {
    fn from(a:  [f64; 3]) -> Primitive {
        Primitive(a[0], a[1], a[2])    
    }
}




// ============================================================================
impl Conserved {
    pub fn density        (self)  -> f64 { self.0 }
    pub fn momentum       (self)  -> f64 { self.1 }
    pub fn total_energy   (self)  -> f64 { self.2 }
    pub fn kinetic_energy (self)  -> f64 { self.momentum().powi(2) / (2.0 * self.density()) }
    pub fn thermal_energy (self)  -> f64 { self.total_energy() - self.kinetic_energy() }
    pub fn to_primitive   (self, gamma_law_index: f64) -> Primitive {
        Primitive(
            self.density(),
            self.momentum() / self.density(),
            self.thermal_energy() * (gamma_law_index - 1.0))
    }
}




// ============================================================================
impl Primitive {
    pub fn density (self) -> f64 { self.0 }
    pub fn velocity(self) -> f64 { self.1 }
    pub fn pressure(self) -> f64 { self.2 }

    pub fn momentum(self) -> f64 {
        self.density() * self.velocity()
    }

    pub fn total_energy(self, gamma_law_index: f64) -> f64 {
        let kinetic = 0.5 * self.density() * self.velocity().powi(2);
        let thermal = self.pressure() / (gamma_law_index - 1.0);
        kinetic + thermal
    }

    pub fn sound_speed_squared(self, gamma_law_index: f64) -> f64 {
        gamma_law_index * self.pressure() / self.density()
    }

    pub fn to_conserved(self, gamma_law_index: f64) -> Conserved {
        Conserved(
            self.density(),
            self.momentum(),
            self.total_energy(gamma_law_index))
    }

    pub fn outer_wavespeeds(self, gamma_law_index: f64) -> (f64, f64) {
        let cs = self.sound_speed_squared(gamma_law_index).sqrt();
        let vn = self.velocity();
        (vn - cs, vn + cs)
    }

    pub fn flux_vector(self, gamma_law_index: f64) -> Conserved {
        let pg = self.pressure();
        let vn = self.velocity();
        let advective_term = vn * self.to_conserved(gamma_law_index);
        let pressure_term = Conserved(0.0, pg, pg * vn);
        advective_term + pressure_term
    }
}




// ============================================================================
pub fn riemann_hlle(pl: Primitive, pr: Primitive, gamma_law_index: f64) -> Conserved {
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
