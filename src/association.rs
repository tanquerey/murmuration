use crate::measurement::{MeasurementModel, PositionSensor};
use crate::motion::{ConstantVelocityModel, MotionModel};
use crate::track::Track;
use nalgebra::Vector3;

pub struct AssociationResult {
    pub matches: Vec<(usize, usize)>, // (track index, detection index)
    pub unmatched_detections: Vec<usize>,
    pub unmatched_tracks: Vec<usize>,
}

pub fn nearest_neighbor_association(
    tracks: &[Track<ConstantVelocityModel, PositionSensor>],
    detections: &[Vector3<f64>],
    gate_distance: f64,
) -> AssociationResult {
    let mut used = vec![false; detections.len()];
    let mut matches = Vec::new();
    let mut unmatched_tracks = Vec::new();

    for (t_idx, track) in tracks.iter().enumerate() {
        let predicted_pos = track.position(); // "where do I expect to be, right now?"

        // Find the closest not-yet-claimed detection, if any is within the gate.
        let mut best: Option<(usize, f64)> = None;
        for (d_idx, det) in detections.iter().enumerate() {
            if used[d_idx] {
                continue;
            }
            let distance = (det - predicted_pos).norm();
            if distance < gate_distance {
                if best.map_or(true, |(_, best_dist)| distance < best_dist) {
                    best = Some((d_idx, distance));
                }
            }
        }

        match best {
            Some((d_idx, _)) => {
                used[d_idx] = true;
                matches.push((t_idx, d_idx));
            }
            None => unmatched_tracks.push(t_idx), // nothing nearby — track goes unseen this frame
        }
    }

    let unmatched_detections = used
        .iter()
        .enumerate()
        .filter_map(|(i, &is_used)| if is_used { None } else { Some(i) })
        .collect();

    AssociationResult {
        matches,
        unmatched_detections,
        unmatched_tracks,
    }
}
