use nalgebra::{SMatrix, SVector};

pub type State = SVector<f64, 6>;
pub type Covariance = SMatrix<f64, 6, 6>;

pub mod motion;
pub mod measurement;
pub mod track;
pub mod association;