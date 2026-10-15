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
    let noise_std = 0.02; // radians, ~1.1 degrees

    let sensor_position1 = Vector3::new(0.0, 0.0, 0.0);
    let sensor_position2 = Vector3::new(80.0, 0.0, 0.0);
    let sensor1 = BearingSensor::new(sensor_position1, noise_std);
    let sensor2 = BearingSensor::new(sensor_position2, noise_std);

    let mut bearing_track = Track::new(
        1,
        State::new(35.0, -60.0, 0.0, 0.0, 5.0, 0.0),
        Covariance::from_diagonal(&SVector::<f64, 6>::new(
            400.0, 400.0, 400.0, 25.0, 25.0, 25.0,
        )),
        ConstantVelocityModel::new(0.05),
    );

    let mut rng = StdRng::seed_from_u64(42);

    for step in 0..30 {
        true_position += true_velocity * dt;

        let noisy1 =
            bearing(sensor_position1, true_position) + rng.gen_range(-noise_std..noise_std);
        let noisy2 =
            bearing(sensor_position2, true_position) + rng.gen_range(-noise_std..noise_std);

        bearing_track.predict(dt);
        bearing_track.update(&sensor1, DVector::from_vec(vec![noisy1]));
        bearing_track.update(&sensor2, DVector::from_vec(vec![noisy2]));

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

fn bearing(from: Vector3<f64>, to: Vector3<f64>) -> f64 {
    (to.y - from.y).atan2(to.x - from.x)
}
