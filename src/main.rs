use nalgebra::{SMatrix, SVector, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// State: [x, y, z, vx, vy, vz]
type State = SVector<f64, 6>;
type Covariance = SMatrix<f64, 6, 6>;

struct KalmanFilter {
    state: State,
    covariance: Covariance,
    process_noise: Covariance,
    measurement_noise: SMatrix<f64, 3, 3>,
}

impl KalmanFilter {
    fn new(state: State, covariance: Covariance) -> Self {
        let q = 0.05;
        let process_noise = Covariance::from_diagonal(&SVector::<f64, 6>::from_element(q));
        let measurement_noise = SMatrix::<f64, 3, 3>::from_diagonal(&Vector3::new(1.0, 1.0, 1.0));
        Self { state, covariance, process_noise, measurement_noise }
    }

    fn transition_matrix(dt: f64) -> Covariance {
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

    fn predict(&mut self, dt: f64) {
        let f = Self::transition_matrix(dt);
        self.state = f * self.state;
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
    }

    fn update(&mut self, measurement: Vector3<f64>) {
        #[rustfmt::skip]
        let h = SMatrix::<f64, 3, 6>::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        );
        let innovation = measurement - h * self.state;
        let s = h * self.covariance * h.transpose() + self.measurement_noise;
        let s_inv = s.try_inverse().expect("innovation covariance should be invertible");
        let k = self.covariance * h.transpose() * s_inv;
        self.state += k * innovation;
        self.covariance = (Covariance::identity() - k * h) * self.covariance;
    }
}

fn main() {
    let dt = 1.0;
    let true_velocity = Vector3::new(5.0, 2.0, 1.0);
    let mut true_position = Vector3::new(0.0, 0.0, 0.0);

    let mut kf = KalmanFilter::new(
        State::zeros(),
        Covariance::from_diagonal(&SVector::<f64, 6>::from_element(100.0)),
    );

    let mut rng = StdRng::seed_from_u64(42);
    let noise_std = 1.0;

    for step in 0..10 {
        true_position += true_velocity * dt;
        let noisy_measurement = Vector3::new(
            true_position.x + rng.gen_range(-noise_std..noise_std),
            true_position.y + rng.gen_range(-noise_std..noise_std),
            true_position.z + rng.gen_range(-noise_std..noise_std),
        );

        kf.predict(dt);
        kf.update(noisy_measurement);

        let err = (kf.state.fixed_rows::<3>(0) - true_position).norm();
        println!(
            "step {step}: true=({:.2},{:.2},{:.2}) est=({:.2},{:.2},{:.2}) err={:.3}",
            true_position.x, true_position.y, true_position.z,
            kf.state[0], kf.state[1], kf.state[2], err
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_toward_true_state_despite_noisy_measurements() {
        let dt = 1.0;
        let true_velocity = Vector3::new(5.0, 2.0, 1.0);
        let mut true_position = Vector3::new(0.0, 0.0, 0.0);
        let mut kf = KalmanFilter::new(
            State::zeros(),
            Covariance::from_diagonal(&SVector::<f64, 6>::from_element(100.0)),
        );
        let mut rng = StdRng::seed_from_u64(7);

        for _ in 0..30 {
            true_position += true_velocity * dt;
            let noisy = Vector3::new(
                true_position.x + rng.gen_range(-1.0..1.0),
                true_position.y + rng.gen_range(-1.0..1.0),
                true_position.z + rng.gen_range(-1.0..1.0),
            );
            kf.predict(dt);
            kf.update(noisy);
        }

        let pos_error = (kf.state.fixed_rows::<3>(0) - true_position).norm();
        assert!(pos_error < 2.0, "position error too large: {pos_error}");
        assert!((kf.state[3] - true_velocity.x).abs() < 1.0);
        assert!((kf.state[4] - true_velocity.y).abs() < 1.0);
        assert!((kf.state[5] - true_velocity.z).abs() < 1.0);
    }
}