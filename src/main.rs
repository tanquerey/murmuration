use murmuration::association::nearest_neighbor_association;
use murmuration::measurement::{BearingSensor, PositionSensor};
use murmuration::motion::ConstantVelocityModel;
use murmuration::track::Track;
use nalgebra::{DVector, SMatrix, SVector, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// State: [x, y, z, vx, vy, vz]
type State = SVector<f64, 6>;
type Covariance = SMatrix<f64, 6, 6>;

fn main() {
    let dt = 1.0;
    let mut true_position = Vector3::new(20.0, -60.0, 0.0);
    let true_velocity = Vector3::new(0.0, 5.0, 0.0);
    let sensor_position = Vector3::new(0.0, 0.0, 0.0);

    let mut bearing_track = Track::new(
        1,
        State::new(35.0, -60.0, 0.0, 0.0, 5.0, 0.0),
        Covariance::from_diagonal(&SVector::<f64, 6>::new(
            400.0, 400.0, 400.0, 25.0, 25.0, 25.0,
        )),
        ConstantVelocityModel::new(0.05),
        BearingSensor::new(sensor_position, 0.02),
    );

    let mut rng = StdRng::seed_from_u64(42);
    let noise_std = 0.02; // radians, ~1.1 degrees

    for step in 0..30 {
        true_position += true_velocity * dt;

        let dx = true_position.x - sensor_position.x;
        let dy = true_position.y - sensor_position.y;
        let true_bearing = dy.atan2(dx);
        let noisy_bearing = true_bearing + rng.gen_range(-noise_std..noise_std);

        bearing_track.predict(dt);
        bearing_track.update(DVector::from_vec(vec![noisy_bearing]));

        let est = bearing_track.position();
        println!(
            "step {step}: true=({:.1},{:.1}) est=({:.1},{:.1}) pos_var=({:.2},{:.2})",
            true_position.x,
            true_position.y,
            est.x,
            est.y,
            bearing_track.covariance[(0, 0)],
            bearing_track.covariance[(1, 1)]
        );
    }
}
