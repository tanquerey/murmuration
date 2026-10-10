use crate::State;
use nalgebra::{DMatrix, DVector, Vector3};

pub trait MeasurementModel {
    /// "What would this sensor expect to see, given my current belief?"
    fn predict_measurement(&self, state: &State) -> DVector<f64>;

    /// Local sensitivity of the measurement to the state, recomputed at `state`
    /// every call. For a linear sensor this comes out the same every time;
    /// for a nonlinear one it genuinely depends on where you currently are.
    fn jacobian(&self, state: &State) -> DMatrix<f64>;

    fn noise_covariance(&self) -> DMatrix<f64>;

    /// Default: plain subtraction. Sensors with wraparound (angles) override this.
    fn innovation(&self, measurement: &DVector<f64>, predicted: &DVector<f64>) -> DVector<f64> {
        measurement - predicted
    }
}

pub struct PositionSensor {
    noise: DMatrix<f64>,
}

impl PositionSensor {
    pub fn new(noise_std: f64) -> Self {
        let var = noise_std * noise_std;
        Self { noise: DMatrix::from_diagonal(&DVector::from_vec(vec![var, var, var])) }
    }
}

impl MeasurementModel for PositionSensor {
    fn predict_measurement(&self, state: &State) -> DVector<f64> {
        DVector::from_vec(vec![state[0], state[1], state[2]])
    }
    fn jacobian(&self, _state: &State) -> DMatrix<f64> {
        let mut h = DMatrix::zeros(3, 6);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;
        h
    }
    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise.clone()
    }
}

pub struct BearingSensor {
    pub sensor_position: Vector3<f64>,
    noise: DMatrix<f64>,
}

impl BearingSensor {
    pub fn new(sensor_position: Vector3<f64>, noise_std: f64) -> Self {
        Self { sensor_position, noise: DMatrix::from_vec(1, 1, vec![noise_std * noise_std]) }
    }
}

impl MeasurementModel for BearingSensor {
    fn predict_measurement(&self, state: &State) -> DVector<f64> {
        let dx = state[0] - self.sensor_position.x;
        let dy = state[1] - self.sensor_position.y;
        DVector::from_vec(vec![dy.atan2(dx)])
    }
    fn jacobian(&self, state: &State) -> DMatrix<f64> {
        let dx = state[0] - self.sensor_position.x;
        let dy = state[1] - self.sensor_position.y;
        let r2 = dx * dx + dy * dy;
        DMatrix::from_row_slice(1, 6, &[-dy / r2, dx / r2, 0.0, 0.0, 0.0, 0.0])
    }
    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise.clone()
    }
    fn innovation(&self, measurement: &DVector<f64>, predicted: &DVector<f64>) -> DVector<f64> {
        let mut diff = measurement - predicted;
        while diff[0] > std::f64::consts::PI { diff[0] -= 2.0 * std::f64::consts::PI; }
        while diff[0] < -std::f64::consts::PI { diff[0] += 2.0 * std::f64::consts::PI; }
        diff
    }
}