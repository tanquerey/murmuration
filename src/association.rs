use crate::gating::{chi_square_99, mahalanobis_sq, Detection};
use crate::motion::MotionModel;
use crate::track::Track;

pub struct AssociationResult {
    pub matches: Vec<(usize, usize)>, // (track index, detection index)
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

/// Greedy global association. Tracks must already be predicted to the
/// detections' time.
///
/// 1. Score every track-detection pair with its squared Mahalanobis distance.
/// 2. Keep only pairs inside the gate (below the chi-square cutoff).
/// 3. Take pairs cheapest-first across all tracks, skipping any pair whose
///    track or detection is already taken.
///
/// This removes the old per-track ordering bias, but it is still greedy, so it
/// can lose to a globally optimal assignment (see the ignored test).
pub fn greedy_association<M: MotionModel>(
    tracks: &[Track<M>],
    detections: &[Detection<'_>],
) -> AssociationResult {
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (t_idx, track) in tracks.iter().enumerate() {
        for (d_idx, det) in detections.iter().enumerate() {
            let cost = mahalanobis_sq(track, det);
            if cost < chi_square_99(det.measurement.len()) {
                candidates.push((cost, t_idx, d_idx));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut track_used = vec![false; tracks.len()];
    let mut det_used = vec![false; detections.len()];
    let mut matches = Vec::new();
    for (_, t_idx, d_idx) in candidates {
        if !track_used[t_idx] && !det_used[d_idx] {
            track_used[t_idx] = true;
            det_used[d_idx] = true;
            matches.push((t_idx, d_idx));
        }
    }

    let unmatched_tracks = track_used
        .iter()
        .enumerate()
        .filter_map(|(i, &used)| if used { None } else { Some(i) })
        .collect();
    let unmatched_detections = det_used
        .iter()
        .enumerate()
        .filter_map(|(i, &used)| if used { None } else { Some(i) })
        .collect();

    AssociationResult { matches, unmatched_tracks, unmatched_detections }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::PositionSensor;
    use crate::motion::ConstantVelocityModel;
    use crate::{Covariance, State};
    use nalgebra::{DVector, SVector};

    fn track(id: usize, x: f64, pos_var: f64) -> Track<ConstantVelocityModel> {
        Track::new(
            id,
            State::new(x, 0.0, 0.0, 0.0, 0.0, 0.0),
            Covariance::from_diagonal(&SVector::<f64, 6>::from([
                pos_var, pos_var, pos_var, 1.0, 1.0, 1.0,
            ])),
            ConstantVelocityModel::new(0.05),
        )
    }

    fn detection_at_x(sensor: &PositionSensor, x: f64) -> Detection<'_> {
        Detection { sensor, measurement: DVector::from_vec(vec![x, 0.0, 0.0]) }
    }

    #[test]
    fn shuffled_detections_go_to_the_nearest_tracks() {
        let sensor = PositionSensor::new(0.0);
        let tracks = vec![track(0, 0.0, 4.0), track(1, 50.0, 4.0)];
        // deliberately listed in the opposite order to the tracks
        let detections = vec![detection_at_x(&sensor, 51.0), detection_at_x(&sensor, 1.0)];

        let result = greedy_association(&tracks, &detections);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 1)));
        assert!(result.matches.contains(&(1, 0)));
        assert!(result.unmatched_tracks.is_empty());
        assert!(result.unmatched_detections.is_empty());
    }

    #[test]
    fn detection_outside_every_gate_stays_unmatched() {
        let sensor = PositionSensor::new(0.0);
        let tracks = vec![track(0, 0.0, 4.0)];
        let detections = vec![detection_at_x(&sensor, 20.0)];

        let result = greedy_association(&tracks, &detections);

        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0]);
        assert_eq!(result.unmatched_detections, vec![0]);
    }

    #[test]
    fn gate_width_follows_each_tracks_own_uncertainty() {
        let sensor = PositionSensor::new(0.0);
        let detections = vec![detection_at_x(&sensor, 6.0)]; // 6 units away from both

        let confident = vec![track(0, 0.0, 1.0)]; // d² = 36 / 1  = 36
        let uncertain = vec![track(1, 0.0, 25.0)]; // d² = 36 / 25 = 1.44

        assert!(greedy_association(&confident, &detections).matches.is_empty());
        assert_eq!(greedy_association(&uncertain, &detections).matches, vec![(0, 0)]);
    }

    // Acceptance test for the Hungarian step. Greedy takes the cheapest pair
    // (T0-D0, cost 4.0) first, which leaves T1 with nothing, so only 1 match.
    // The best assignment is T0-D1 (10.56) + T1-D0 (9.0), which matches both.
    #[test]
    #[ignore = "needs Hungarian assignment"]
    fn contested_detection_goes_to_the_globally_best_pairing() {
        let sensor = PositionSensor::new(0.0);
        let tracks = vec![track(0, 0.0, 4.0), track(1, 10.0, 4.0)];
        let detections = vec![detection_at_x(&sensor, 4.0), detection_at_x(&sensor, -6.5)];

        let result = greedy_association(&tracks, &detections);

        assert_eq!(result.matches.len(), 2);
        assert!(result.matches.contains(&(0, 1)));
        assert!(result.matches.contains(&(1, 0)));
    }
}