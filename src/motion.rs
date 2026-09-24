use nalgebra::{SMatrix, SVector};

use crate::Covariance;

pub trait MotionModel {
    fn transition_matrix(&self, dt: f64) -> Covariance;
    fn process_noise(&self) -> &Covariance;
}

pub struct ConstantVelocityModel {
    process_noise: Covariance,
}

impl ConstantVelocityModel {
    pub fn new(q: f64) -> Self {
        Self { process_noise: Covariance::from_diagonal(&SVector::<f64, 6>::from_element(q)) }
    }
}

impl MotionModel for ConstantVelocityModel {
    fn transition_matrix(&self, dt: f64) -> Covariance {
        #[rustfmt::skip]
        let f = SMatrix::<f64, 6, 6>::new(
            1.0, 0.0, 0.0, dt,  0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, dt,  0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, dt,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        f
    }
    fn process_noise(&self) -> &Covariance {
        &self.process_noise
    }
}