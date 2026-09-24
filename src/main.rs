use murmuration::association::nearest_neighbor_association;
use murmuration::measurement::PositionSensor;
use murmuration::motion::ConstantVelocityModel;
use murmuration::track::Track;
use nalgebra::{SMatrix, SVector, Vector3};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

// State: [x, y, z, vx, vy, vz]
type State = SVector<f64, 6>;
type Covariance = SMatrix<f64, 6, 6>;

fn main() {
    let dt = 1.0;
    // Two real objects, on different paths.
    let mut true_positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(100.0, 0.0, 0.0)];
    let true_velocities = vec![Vector3::new(5.0, 2.0, 1.0), Vector3::new(-3.0, 4.0, 0.0)];

    // Tracks initialized at truth for now — proper track *creation* from scratch is a later step.
    let mut tracks = vec![
        Track::new(
            0,
            State::new(0.0, 0.0, 0.0, 5.0, 2.0, 1.0),
            Covariance::from_diagonal(&SVector::from_element(10.0)),
            ConstantVelocityModel::new(0.05),   // motion_model
            PositionSensor::new(1.0),           // measurement_model
        ),
        Track::new(
            1,
            State::new(100.0, 0.0, 0.0, -3.0, 4.0, 0.0),
            Covariance::from_diagonal(&SVector::from_element(10.0)),
            ConstantVelocityModel::new(0.05),
            PositionSensor::new(1.0),
        ),
    ];

    let mut rng = StdRng::seed_from_u64(42);

    for step in 0..10 {
        for i in 0..2 {
            true_positions[i] += true_velocities[i] * dt;
        }

        // Sensor hands back noisy detections in ARBITRARY order — the whole point of this step.
        let mut detections: Vec<Vector3<f64>> = true_positions
            .iter()
            .map(|p| Vector3::new(
                p.x + rng.gen_range(-1.0..1.0),
                p.y + rng.gen_range(-1.0..1.0),
                p.z + rng.gen_range(-1.0..1.0),
            ))
            .collect();
        detections.shuffle(&mut rng);

        for track in &mut tracks {
            track.predict(dt);
        }

        let result = nearest_neighbor_association(&tracks, &detections, 15.0);

        for (t_idx, d_idx) in &result.matches {
            tracks[*t_idx].update(detections[*d_idx]);
        }

        println!("step {step}: {} matched, {} unmatched detections, {} unmatched tracks",
            result.matches.len(), result.unmatched_detections.len(), result.unmatched_tracks.len());
        for track in &tracks {
            println!("  track {}: est={:?}", track.id, track.position());
        }
    }
}