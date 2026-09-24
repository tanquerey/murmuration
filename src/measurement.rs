use nalgebra::{SMatrix, Vector3};

pub trait MeasurementModel {
    fn h(&self) -> SMatrix<f64, 3, 6>;
    fn noise_covariance(&self) -> &SMatrix<f64, 3, 3>;
}

pub struct PositionSensor {
    noise: SMatrix<f64, 3, 3>,
}

impl PositionSensor {
    pub fn new(noise_std: f64) -> Self {
        Self { noise: SMatrix::from_diagonal(&Vector3::new(noise_std, noise_std, noise_std)) }
    }
}

impl MeasurementModel for PositionSensor {
    fn h(&self) -> SMatrix<f64, 3, 6> {
        #[rustfmt::skip]
        let h = SMatrix::<f64, 3, 6>::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        );
        h
    }
    fn noise_covariance(&self) -> &SMatrix<f64, 3, 3> {
        &self.noise
    }
}