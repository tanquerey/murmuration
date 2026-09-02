use nalgebra::{SMatrix, SVector, Vector2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// State: [x, y, vx, vy]. Fixed-size (not DVector) since dimension is known at compile time.
type State = SVector<f64, 4>;
type Covariance = SMatrix<f64, 4, 4>;

struct KalmanFilter {
    state: State,
    covariance: Covariance,
    process_noise: Covariance,           // Q — how much we distrust constant-velocity
    measurement_noise: SMatrix<f64, 2, 2>, // R — sensor's own noise
}

impl KalmanFilter {
    fn new(state: State, covariance: Covariance) -> Self {
        let q = 0.05;
        let process_noise = Covariance::from_diagonal(&SVector::<f64, 4>::new(q, q, q, q));
        let measurement_noise =
            SMatrix::<f64, 2, 2>::from_diagonal(&Vector2::new(1.0, 1.0));
        Self { state, covariance, process_noise, measurement_noise }
    }

    // Constant-velocity transition matrix: position += velocity * dt, velocity unchanged.
    fn transition_matrix(dt: f64) -> Covariance {
        SMatrix::<f64, 4, 4>::new(
            1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    fn predict(&mut self, dt: f64) {
        let f = Self::transition_matrix(dt);
        self.state = f * self.state;
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
    }

    fn update(&mut self, measurement: Vector2<f64>) {
        // H: we only directly observe position (x, y), not velocity.
        let h = SMatrix::<f64, 2, 4>::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        );
        let innovation = measurement - h * self.state;
        let s = h * self.covariance * h.transpose() + self.measurement_noise;
        let s_inv = s.try_inverse().expect("innovation covariance should be invertible");
        let k = self.covariance * h.transpose() * s_inv; // Kalman gain
        self.state += k * innovation;
        self.covariance = (Covariance::identity() - k * h) * self.covariance;
    }
}

fn main() {
    let dt = 1.0;
    let true_velocity = Vector2::new(5.0, 2.0);
    let mut true_position = Vector2::new(0.0, 0.0);

    let mut kf = KalmanFilter::new(
        State::new(0.0, 0.0, 0.0, 0.0),
        Covariance::from_diagonal(&SVector::<f64, 4>::new(100.0, 100.0, 100.0, 100.0)),
    );

    let mut rng = StdRng::seed_from_u64(42); // fixed seed: deterministic, repeatable runs
    let noise_std = 1.0;

    for step in 0..10 {
        true_position += true_velocity * dt;
        let noisy_measurement = Vector2::new(
            true_position.x + rng.gen_range(-noise_std..noise_std),
            true_position.y + rng.gen_range(-noise_std..noise_std),
        );

        kf.predict(dt);
        kf.update(noisy_measurement);

        let pos_error = (kf.state.x - true_position.x, kf.state.y - true_position.y);
        let pos_error = (pos_error.0.powi(2) + pos_error.1.powi(2)).sqrt();
        println!(
            "step {step}: true=({:.2},{:.2}) est=({:.2},{:.2}) vel_est=({:.2},{:.2}) err={:.3}",
            true_position.x, true_position.y, kf.state.x, kf.state.y, kf.state.z, kf.state.w, pos_error
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_toward_true_state_despite_noisy_measurements() {
        let dt = 1.0;
        let true_velocity = Vector2::new(5.0, 2.0);
        let mut true_position = Vector2::new(0.0, 0.0);
        let mut kf = KalmanFilter::new(
            State::new(0.0, 0.0, 0.0, 0.0),
            Covariance::from_diagonal(&SVector::<f64, 4>::new(100.0, 100.0, 100.0, 100.0)),
        );
        let mut rng = StdRng::seed_from_u64(7);

        for _ in 0..30 {
            true_position += true_velocity * dt;
            let noisy = Vector2::new(
                true_position.x + rng.gen_range(-1.0..1.0),
                true_position.y + rng.gen_range(-1.0..1.0),
            );
            kf.predict(dt);
            kf.update(noisy);
        }

        let pos_error =
            ((kf.state.x - true_position.x).powi(2) + (kf.state.y - true_position.y).powi(2)).sqrt();
        assert!(pos_error < 2.0, "position error too large: {pos_error}");
        assert!((kf.state.z - true_velocity.x).abs() < 1.0, "vx estimate off: {}", kf.state.z);
        assert!((kf.state.w - true_velocity.y).abs() < 1.0, "vy estimate off: {}", kf.state.w);
    }
}