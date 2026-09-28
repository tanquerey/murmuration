use murmuration::association::nearest_neighbor_association;
use murmuration::measurement::PositionSensor;
use murmuration::motion::ConstantVelocityModel;
use murmuration::track::Track;
use nalgebra::{RealField, SMatrix, SVector, Vector3};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

// State: [x, y, z, vx, vy, vz]
type State = SVector<f64, 6>;
type Covariance = SMatrix<f64, 6, 6>;

fn main() {
    let dt = 1.0;
    let mut true_position = Vector3::new(20.0, 0.0, 0.0);
    let true_velocity = Vector3::new(5.0, 2.0, 1.0);
    let sensor_position = Vector3::new(0.0, 0.0, 0.0);

    let mut track = Track::new(
        0,
        State::new(35.0, 0.0, 0.0, 5.0, 2.0, 1.0),
        Covariance::from_diagonal(&SVector::from_element(10.0)),
        ConstantVelocityModel::new(0.05),
        PositionSensor::new(1.0),
    );

    let mut rng = StdRng::seed_from_u64(42);
    let noise_std = 0.02; // radians, ~1.1 degrees

    for step in 0..10 {
        true_position += true_velocity * dt;

        let dx = true_position.x - sensor_position.x;
        let dy = true_position.y - sensor_position.y;
        let true_bearing = dy.atan2(dx);
        let noisy_bearing = true_bearing + rng.gen_range(-noise_std..noise_std);

        track.predict(dt);
        track.update_bearing(sensor_position, noisy_bearing, noise_std * noise_std);

        let est = track.position();
        println!(
            "step {step}: true=({:.1},{:.1}) est=({:.1},{:.1}) pos_var=({:.2},{:.2})",
            true_position.x, true_position.y, est.x, est.y,
            track.covariance[(0, 0)], track.covariance[(1, 1)]
        );
    }
}