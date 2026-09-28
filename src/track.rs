use crate::measurement::MeasurementModel;
use crate::motion::MotionModel;
use crate::{Covariance, State};
use nalgebra::{SMatrix, Vector3};

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

    pub fn update(&mut self, measurement: Vector3<f64>) {
        let h = self.measurement_model.h();
        let r = self.measurement_model.noise_covariance();
        let innovation = measurement - h * self.state;
        let s = h * self.covariance * h.transpose() + r;
        let s_inv = s
            .try_inverse()
            .expect("innovation covariance should be invertible");
        let k = self.covariance * h.transpose() * s_inv;
        self.state += k * innovation;
        self.covariance = (Covariance::identity() - k * h) * self.covariance;
    }

    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }
    
    pub fn update_bearing(&mut self, sensor_position: Vector3<f64>, measured_bearing: f64, bearing_noise_var: f64) {
        let dx = self.state[0] - sensor_position.x;
        let dy = self.state[1] - sensor_position.y;
        let r2 = dx * dx + dy * dy;

        // "what angle would I expect, given my current belief?" — the atan2 formula, not a fixed H.
        let predicted_bearing = dy.atan2(dx);

        let mut innovation = measured_bearing - predicted_bearing;
        // angles wrap: -179° and +179° are 2° apart, not 358° — normalize into [-π, π]
        while innovation > std::f64::consts::PI { innovation -= 2.0 * std::f64::consts::PI; }
        while innovation < -std::f64::consts::PI { innovation += 2.0 * std::f64::consts::PI; }

        // the Jacobian — local sensitivity of bearing to (x, y), recomputed every call
        #[rustfmt::skip]
        let h = SMatrix::<f64, 1, 6>::new(-dy / r2, dx / r2, 0.0, 0.0, 0.0, 0.0);

        let s = (h * self.covariance * h.transpose())[(0, 0)] + bearing_noise_var;
        let k = (self.covariance * h.transpose()) / s; // 6x1 — a plain scalar division this time
        self.state += k * innovation;
        self.covariance = (Covariance::identity() - k * h) * self.covariance;
    }

}

// #[cfg(test)]
// mod tests {
//     use nalgebra::SVector;

// use super::*;

//     #[test]
//     fn converges_toward_true_state_despite_noisy_measurements() {
//         let dt = 1.0;
//         let true_velocity = Vector3::new(5.0, 2.0, 1.0);
//         let mut true_position = Vector3::new(0.0, 0.0, 0.0);
//         let mut kf = Track::new(
//             State::zeros(),
//             Covariance::from_diagonal(&SVector::<f64, 6>::from_element(100.0)),
//         );
//         let mut rng = StdRng::seed_from_u64(7);

//         for _ in 0..30 {
//             true_position += true_velocity * dt;
//             let noisy = Vector3::new(
//                 true_position.x + rng.gen_range(-1.0..1.0),
//                 true_position.y + rng.gen_range(-1.0..1.0),
//                 true_position.z + rng.gen_range(-1.0..1.0),
//             );
//             kf.predict(dt);
//             kf.update(noisy);
//         }

//         let pos_error = (kf.state.fixed_rows::<3>(0) - true_position).norm();
//         assert!(pos_error < 2.0, "position error too large: {pos_error}");
//         assert!((kf.state[3] - true_velocity.x).abs() < 1.0);
//         assert!((kf.state[4] - true_velocity.y).abs() < 1.0);
//         assert!((kf.state[5] - true_velocity.z).abs() < 1.0);
//     }
// }
