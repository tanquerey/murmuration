use murmuration::association::greedy_association;
use murmuration::gating::Detection;
use murmuration::measurement::PositionSensor;
use murmuration::motion::ConstantVelocityModel;
use murmuration::track::Track;
use murmuration::{Covariance, State};
use nalgebra::{DVector, SVector, Vector3};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn main() {
    let dt = 1.0;
    let mut truth = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(100.0, 0.0, 0.0)];
    let velocity = vec![Vector3::new(5.0, 2.0, 1.0), Vector3::new(-3.0, 4.0, 0.0)];
    let sensor = PositionSensor::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    let mut tracks: Vec<Track<ConstantVelocityModel>> = (0..2)
        .map(|i| {
            Track::new(
                i,
                State::new(
                    truth[i].x, truth[i].y, truth[i].z,
                    velocity[i].x, velocity[i].y, velocity[i].z,
                ),
                Covariance::from_diagonal(&SVector::<f64, 6>::from_element(10.0)),
                ConstantVelocityModel::new(0.05),
            )
        })
        .collect();

    for step in 0..10 {
        for i in 0..2 {
            truth[i] += velocity[i] * dt;
        }

        let mut readings: Vec<DVector<f64>> = truth
            .iter()
            .map(|p| {
                DVector::from_vec(vec![
                    p.x + rng.gen_range(-1.0..1.0),
                    p.y + rng.gen_range(-1.0..1.0),
                    p.z + rng.gen_range(-1.0..1.0),
                ])
            })
            .collect();
        readings.shuffle(&mut rng); // the sensor gives no labels and no ordering

        let detections: Vec<Detection<'_>> = readings
            .into_iter()
            .map(|measurement| Detection { sensor: &sensor, measurement })
            .collect();

        for track in &mut tracks {
            track.predict(dt);
        }
        let result = greedy_association(&tracks, &detections);
        for &(t, d) in &result.matches {
            tracks[t].update(&sensor, detections[d].measurement.clone());
        }

        println!(
            "step {step}: {} matched, {} unmatched tracks, {} unmatched detections",
            result.matches.len(),
            result.unmatched_tracks.len(),
            result.unmatched_detections.len()
        );
    }
}