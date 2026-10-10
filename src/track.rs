use crate::measurement::MeasurementModel;
use crate::motion::MotionModel;
use crate::{Covariance, State};
use nalgebra::{DMatrix, Vector3};

pub struct Track<M: MotionModel, Z: MeasurementModel> {
    pub id: usize,
    pub state: State,
    pub covariance: Covariance,
    pub motion_model: M,
    pub measurement_model: Z,
}

impl<M: MotionModel, Z: MeasurementModel> Track<M, Z> {
    pub fn new(
        id: usize,
        state: State,
        covariance: Covariance,
        motion_model: M,
        measurement_model: Z,
    ) -> Self {
        Self {
            id,
            state,
            covariance,
            motion_model,
            measurement_model,
        }
    }

    pub fn predict(&mut self, dt: f64) {
        let f = self.motion_model.transition_matrix(dt);
        self.state = f * self.state;
        self.covariance = f * self.covariance * f.transpose() + self.motion_model.process_noise();
    }

pub fn update(&mut self, measurement: nalgebra::DVector<f64>) {
    let predicted = self.measurement_model.predict_measurement(&self.state);
    let h = self.measurement_model.jacobian(&self.state);
    let r = self.measurement_model.noise_covariance();
    let innovation = self.measurement_model.innovation(&measurement, &predicted);

    // bridge: fixed-size state/covariance -> dynamic, to match the sensor's own dimension
    let cov = DMatrix::from_column_slice(6, 6, self.covariance.as_slice());

    let s = &h * &cov * h.transpose() + &r;
    let s_inv = s.clone().try_inverse().expect("S should be invertible");
    let k = &cov * h.transpose() * s_inv;

    let state_update = &k * innovation;
    for i in 0..6 {
        self.state[i] += state_update[i];
    }

    // Joseph form — numerically safer than (I - KH) * P, same reasoning as the blow-up you saw earlier
    let identity = DMatrix::identity(6, 6);
    let i_minus_kh = &identity - &k * &h;
    let new_cov = &i_minus_kh * &cov * i_minus_kh.transpose() + &k * &r * k.transpose();
    for i in 0..6 {
        for j in 0..6 {
            self.covariance[(i, j)] = new_cov[(i, j)];
        }
    }
}

    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }   
}
