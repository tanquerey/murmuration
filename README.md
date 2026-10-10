# murmuration

A multi-sensor **track fusion and correlation engine** written in Rust, aiming for production-grade quality. It targets the core hard problem underneath any counter-UAS situational-awareness / Common Operational Picture (COP) system.

> Named after starling murmurations: many independent, noisy observers whose local signals combine into one coherent, emergent picture. That is what sensor fusion does with raw detections.

**Status: early prototype.** There is a 3D constant-velocity Kalman filter with swappable measurement models (position and bearing-only), greedy nearest-neighbor association, and simulated-scenario experiments. It is not yet a complete fusion engine. See the roadmap and open questions below.

---

## Why this project exists

The long-term goal is a full COP system that ingests feeds from heterogeneous drones and sensors, fuses them into a live shared picture, and assigns interceptors to threats. Building all of that at once would produce something broad but thin.

Instead, this project scopes down to **one piece, built as deep and correct as possible**: multi-sensor track fusion. It is the hardest part of a COP, and it needs real understanding of state estimation, uncertainty, and data association. The design uses traits as extension points so it can grow toward a fuller system later.

Scope is limited to the **air/drone domain**. Ground and air targets have different motion characteristics, sensor suites, and clutter models, and mixing them adds complexity without adding to the core problem.

---

## Core concepts

- **Detection**: a single raw, noisy reading from one sensor at one instant.
- **Track**: the maintained belief about one real-world object over time, made of a state estimate, an uncertainty (covariance), and an identity. It is built by predicting forward and correcting against incoming detections.
- **State and covariance**: the state is `[x, y, z, vx, vy, vz]`. The covariance describes how uncertain that estimate is, including in which directions. This matters for sensors like bearing-only direction finders, whose uncertainty is a narrow cone (tight in angle, unbounded in range). A single confidence scalar cannot represent that shape.
- **Motion model**: how state evolves between updates (currently constant velocity). Object-specific unpredictability is encoded in process noise (`Q`), not in the model's shape.
- **Measurement model**: how a sensor's native measurement maps to and from the track state. For a linear sensor this is a fixed matrix. For a nonlinear one (bearing, and later range) it is a function plus its Jacobian, recomputed at the current estimate. This is the extended Kalman filter approach.
- **Data association**: deciding which detections belong to which tracks each frame. Currently greedy nearest-neighbor with a fixed Euclidean gate. Mahalanobis gating and global assignment (Hungarian) are planned, with JPDA after that.
- **Track lifecycle**: tentative, confirmed, coasting, dropped. Planned, not implemented.

---

## Why sensors don't just report position

Most sensors measure whatever their physics allows, and position has to be derived:

- **Radar**: round-trip pulse timing gives range, antenna angle gives bearing.
- **Passive EO/IR camera**: angle only; depth needs triangulation or extra cues.
- **RF direction-finding**: bearing only, from phase differences across an antenna array.
- **GPS**: only for cooperative objects that broadcast their own fix. Not available for uncooperative targets, which is the primary case here.

Different sensors contribute differently shaped partial information, and the covariance is what lets the math combine them correctly.

---

## Architecture (as implemented)

```
src/
  lib.rs          State / Covariance type aliases, module declarations
  motion.rs       MotionModel trait, ConstantVelocityModel
  measurement.rs  MeasurementModel trait, PositionSensor, BearingSensor
  track.rs        Track<M, Z>: predict() and a single generic update()
  association.rs  nearest_neighbor_association
  main.rs         simulation harness (edit the scenario here)
```

```rust
pub trait MotionModel {
    fn transition_matrix(&self, dt: f64) -> Covariance;
    fn process_noise(&self) -> &Covariance;
}

pub trait MeasurementModel {
    fn predict_measurement(&self, state: &State) -> DVector<f64>;
    fn jacobian(&self, state: &State) -> DMatrix<f64>;
    fn noise_covariance(&self) -> DMatrix<f64>;
    // default: plain subtraction; BearingSensor overrides it to wrap angles
    fn innovation(&self, measurement: &DVector<f64>, predicted: &DVector<f64>) -> DVector<f64>;
}
```

- `PositionSensor` is linear, with a constant Jacobian.
- `BearingSensor` is nonlinear (azimuth only), with a Jacobian recomputed at each call and an angle-wrapped innovation.
- Both go through the same `Track::update`. The trait boundary held when the second sensor type was added.
- Track state is fixed-size (`SVector<f64, 6>`); measurements are dynamic-size (`DVector`), because dimension differs per sensor (3 for position, 1 for bearing).
- The covariance update uses the Joseph form for numerical robustness.

**Not yet built:** an `AssociationStrategy` trait and a `FusionEngine` that owns tracks and runs the full loop.

---

## Experiments: bearing-only observability

A bearing sensor measures angle but not range, so whether range can be learned depends on geometry. Two runs, same sensor (at the origin, declared noise 0.02 rad), same target speed (5 units/step along a line at x = 20), same filter prior (correct velocity, 15-unit error in x, position variance 400 per axis):

| | Receding target | Target passing at closest approach |
|---|---|---|
| Estimated range / true range | stays roughly 1.1x to 1.4x for all 30 steps | settles around 1.1x |
| y variance | grows to about 6000 | falls from 303 to 0.15 at closest approach, then grows again |
| x error | mostly 5 to 8 units | about 2 to 4 units after closest approach, in line with the filter's own reported spread (about 3) |
| y error at step 29 | 65 | 12.6 |

Takeaways:

- Range is inferred from how fast the bearing changes, which depends on the sideways component of motion divided by distance. As a target recedes, that signal decays and the information disappears.
- The covariance reports this honestly. The filter's uncertainty grew in exactly the direction the sensor cannot see.
- Caveats: these are single runs with one RNG seed. A jump in the receding run is larger than sensor noise explains; the suspected cause is EKF linearization error under large position uncertainty, which has not been verified.

---

## Open design questions

- **Multi-sensor fusion into one track.** A `Track<M, Z>` currently owns one measurement model, so a single track cannot yet be updated by different sensors. The likely fix is passing the sensor's model into `update()` per detection instead of storing it on the track. This is the main gap between the current code and the project's headline goal.
- **Static vs dynamic dispatch.** Generics give static dispatch, but force every track in a `Vec` to share concrete model types.
- **EKF limits for bearing-only tracking.** Alternatives (different state parametrizations, unscented transform, particle filters) are worth evaluating once the pipeline is complete.

---

## Roadmap

1. ✅ Single-object, single-sensor Kalman filter
2. ✅ 3D state
3. ✅ Multiple objects with greedy nearest-neighbor association and a fixed Euclidean gate
4. ✅ `MotionModel` / `MeasurementModel` traits extracted from working code
5. ✅ Second sensor type (bearing-only) through the unified `Track::update`. Still to do: rerun the position-sensor scenarios as a regression check and restore the unit tests.
6. 🔜 Per-detection measurement models, so one track can fuse several sensors
7. 🔜 Mahalanobis gating and global nearest neighbor (Hungarian)
8. 🔜 Track lifecycle state machine
9. 🔜 JPDA as a second association strategy, benchmarked against GNN

Longer term, optional: live map/UI, interceptor assignment, exporting fused tracks in a standard interoperable format.

---

## Design principles

- **Depth over breadth.** One correctly solved hard problem beats a shallow full system.
- **Full covariance, always.** Never collapse uncertainty to a single confidence scalar. If a sensor only exposes a 0-1 confidence, use it to scale a known baseline covariance, never to replace one.
- **Traits grow from working code**, not the other way around.
- **Determinism for testability.** Simulated noise uses a fixed RNG seed, and the filter takes an explicit `dt` and never reads the system clock. A timestamped `now` API is planned alongside the lifecycle state machine.
- **Math via `nalgebra`, algorithms hand-written.** Linear algebra primitives are not reinvented; the filtering, association, and fusion logic is the point of the project.


---


## Getting started

```
cargo run
```

This runs the scenario in `src/main.rs`.