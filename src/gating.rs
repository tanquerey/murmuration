use crate::measurement::MeasurementModel;
use crate::motion::MotionModel;
use crate::track::Track;
use nalgebra::{DMatrix, DVector};

/// A reading tagged with the sensor that produced it. A bearing reading
/// means nothing without knowing where that sensor is.
pub struct Detection<'a> {
    pub sensor: &'a dyn MeasurementModel,
    pub measurement: DVector<f64>,
}

/// 99% chi-square cutoffs, indexed by how many numbers are in the reading.
pub fn chi_square_99(dof: usize) -> f64 {
    match dof {
        1 => 6.63,
        2 => 9.21,
        3 => 11.34,
        _ => panic!("no gate cutoff for a {dof}-value reading; extend this table"),
    }
}

/// Squared Mahalanobis distance between a detection and what the track
/// expects that sensor to read. The track must already be predicted to
/// the detection's time.
pub fn mahalanobis_sq<M: MotionModel>(track: &Track<M>, detection: &Detection<'_>) -> f64 {
    let predicted = detection.sensor.predict_measurement(&track.state);
    let h = detection.sensor.jacobian(&track.state);
    let r = detection.sensor.noise_covariance();
    let innovation = detection.sensor.innovation(&detection.measurement, &predicted);

    let cov = DMatrix::from_column_slice(6, 6, track.covariance.as_slice());
    let s = &h * &cov * h.transpose() + &r;
    let Some(s_inv) = s.try_inverse() else {
        return f64::INFINITY; // can't judge plausibility, so call it implausible
    };

    let weighted = &s_inv * &innovation;
    innovation.dot(&weighted)
}

/// Is this detection plausible for this track at the 99% level?
pub fn passes_gate<M: MotionModel>(track: &Track<M>, detection: &Detection<'_>) -> bool {
    mahalanobis_sq(track, detection) < chi_square_99(detection.measurement.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::{BearingSensor, PositionSensor};
    use crate::motion::ConstantVelocityModel;
    use crate::{Covariance, State};
    use nalgebra::{SVector, Vector3};

    fn track_at(state: State, cov_diag: [f64; 6]) -> Track<ConstantVelocityModel> {
        Track::new(
            0,
            state,
            Covariance::from_diagonal(&SVector::<f64, 6>::from(cov_diag)),
            ConstantVelocityModel::new(0.05),
        )
    }

    #[test]
    fn position_gate_matches_hand_calculation() {
        let track = track_at(
            State::new(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            [4.0, 4.0, 4.0, 1.0, 1.0, 1.0],
        );
        let sensor = PositionSensor::new(0.0);
        let a = Detection { sensor: &sensor, measurement: DVector::from_vec(vec![13.0, 1.0, 2.0]) };
        let b = Detection { sensor: &sensor, measurement: DVector::from_vec(vec![16.0, 5.0, 4.0]) };

        assert!((mahalanobis_sq(&track, &a) - 3.5).abs() < 1e-9);
        assert!((mahalanobis_sq(&track, &b) - 19.25).abs() < 1e-9);
        assert!(passes_gate(&track, &a));
        assert!(!passes_gate(&track, &b));
    }

    #[test]
    fn bearing_gate_handles_angle_wraparound() {
        // Track at (20, 0), sensor at the origin: predicted bearing 0, S = 4/400 = 0.01.
        let track = track_at(
            State::new(20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            [4.0, 4.0, 4.0, 1.0, 1.0, 1.0],
        );
        let sensor = BearingSensor::new(Vector3::new(0.0, 0.0, 0.0), 0.0);
        let near = Detection { sensor: &sensor, measurement: DVector::from_vec(vec![0.15]) };
        let far = Detection { sensor: &sensor, measurement: DVector::from_vec(vec![0.30]) };
        let wrapped = Detection {
            sensor: &sensor,
            measurement: DVector::from_vec(vec![std::f64::consts::TAU - 0.1]),
        };

        assert!((mahalanobis_sq(&track, &near) - 2.25).abs() < 1e-9);
        assert!((mahalanobis_sq(&track, &far) - 9.0).abs() < 1e-9);
        assert!((mahalanobis_sq(&track, &wrapped) - 1.0).abs() < 1e-9);
        assert!(passes_gate(&track, &near));
        assert!(!passes_gate(&track, &far));
    }
}