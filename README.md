# murmuration

A production-grade, multi-sensor **track fusion and correlation engine**, written in Rust — the core hard problem underneath any counter-UAS situational-awareness / Common Operational Picture (COP) system.

> Named after starling murmurations: many independent, noisy observers whose local signals combine into one coherent, emergent picture — which is exactly what sensor fusion does with raw detections.

**Status: early prototype.** Core Kalman filter math validated on a single simulated object and sensor.

---

## Why this project exists

The long-term goal is a full COP system to ingest feeds from heterogeneous drones/sensors, fuse them into a live shared picture, and assign interceptors to threats. Building that whole system shallowly, all at once, would produce something broad but thin — every part half-done.

Instead, this project deliberately scopes down to **one piece, built as deep and correct as possible**: multi-sensor track fusion. It's the hardest, most differentiating part of a COP — the part that actually requires understanding state estimation, uncertainty, and data association — and it's designed with extension points (traits) so it can grow toward a fuller system later without a rewrite.

Scope is intentionally limited to the **air/drone domain** (not mixed with ground-based tracking) — ground and air targets have different motion characteristics, sensor suites, and clutter models, and mixing them adds complexity without adding signal to the core problem this project is about.

---

## Core concepts

- **Detection** — a single raw, noisy reading from one sensor at one instant (e.g. "something at this range/bearing, right now").
- **Track** — the maintained belief about one real-world object over time: a state estimate, an uncertainty (covariance), an identity, and a lifecycle status. Built by continuously predicting a track forward and correcting it against incoming detections.
- **State & covariance** — state is the best-guess position/velocity (and eventually orientation) of an object; covariance is a matrix describing *how* uncertain that guess is — not just how much, but in which directions. This matters a lot for sensors like bearing-only RF direction-finders, whose uncertainty is a narrow angular "cone" (tight in angle, unbounded in range) — a shape a single confidence scalar can't represent, and that a full covariance matrix is required to correctly fuse against other sensors.
- **Motion model** — how a track's state evolves between updates (e.g. constant velocity). Object-specific unpredictability (a sluggish object vs. a highly maneuverable drone) is encoded in process noise (`Q`), not in the model's shape.
- **Measurement model** — how a given sensor's native measurement space (range+bearing, bearing-only, pixel coordinates, etc.) maps to and from the track's canonical state space.
- **Data association** — deciding which detections belong to which existing tracks each frame. Starting with Global Nearest Neighbor (GNN, via the Hungarian algorithm) for tractability, with Joint Probabilistic Data Association (JPDA) planned as a second, swappable strategy for handling ambiguous/cluttered scenes.
- **Track lifecycle** — tentative → confirmed → coasting → dropped, governing when a new track is spawned, trusted, and eventually removed after going unseen for too long.

---

## Why sensors don't just report position

Most sensors don't measure lat/long/alt directly — they measure whatever their physical sensing principle allows, and position has to be derived:

- **Radar**: round-trip pulse timing → range, antenna angle → bearing.
- **Passive EO/IR camera**: angle only (pixel direction); depth requires triangulation or additional cues.
- **RF direction-finding**: bearing only, via phase differences across an antenna array.
- **GPS**: only available for *cooperative* objects that receive satellite signals and broadcast their own fix (e.g. a friendly transponder) — not available for uncooperative/adversarial targets, which is the primary case this system is built for.

This is the whole reason fusion is a hard problem rather than simple averaging: different sensors contribute differently-shaped partial information, and the covariance matrix is what lets the math combine them correctly.

---

## Planned architecture (trait design)

The design goal is clean seams so algorithms can be swapped without touching the pipeline:

```rust
trait MotionModel {
    fn predict(&self, state: &DVector<f64>, cov: &DMatrix<f64>, dt: f64)
        -> (DVector<f64>, DMatrix<f64>);
}

trait MeasurementModel {
    fn predict_measurement(&self, state: &DVector<f64>) -> DVector<f64>;
    fn jacobian(&self, state: &DVector<f64>) -> DMatrix<f64>;
    fn noise_covariance(&self) -> &DMatrix<f64>;
}

trait AssociationStrategy {
    fn associate(&self, tracks: &[Track], detections: &[Detection]) -> AssociationResult;
}

struct FusionEngine<M: MotionModel, A: AssociationStrategy> {
    motion_model: M,
    association: A,
    tracks: Vec<Track>,
    gate_threshold: f64,
}
```

**These traits are not implemented yet, by design.** The plan is to extract them from working concrete code once real variation has been felt (e.g. a second sensor type), rather than guessing the right abstraction shape upfront.

---

## Build order / roadmap

1. **Concrete single-object, single-sensor Kalman filter** — no traits, hardcoded constant-velocity model, one simulated sensor. Prove the predict/update math converges correctly. *(current stage — 2D validated, 3D/altitude extension planned next)*
2. **Multiple objects, one sensor** — introduce real `Track`/`Detection` structs, nearest-neighbor/GNN association with gating.
3. **Extract `MotionModel` / `MeasurementModel` traits** — lift working concrete code behind trait boundaries.
4. **Add a second sensor type** (e.g. bearing-only) — the real test of whether the trait boundary is correctly drawn.
5. **Track lifecycle state machine** — tentative/confirmed/coasting/dropped, with deterministic time-based tests.
6. **JPDA as a second `AssociationStrategy`**, benchmarked against GNN on crossing/cluttered scenarios.

Longer-term, optional extensions once the core is solid: live map/UI, interceptor assignment (Hungarian/auction algorithm), and possibly exporting fused tracks in a standard interoperable format.

---

## Design principles

- **Depth over breadth.** One correctly-solved hard problem beats a shallow full system.
- **Full covariance, always.** Never collapse sensor or track uncertainty to a single confidence scalar — if a future sensor only exposes a 0–1 confidence value, it's used to scale a known baseline covariance, never as a replacement for one.
- **Traits grow from working code**, not the other way around.
- **Determinism for testability.** Simulated sensor noise uses a fixed RNG seed; time-dependent logic (prediction, lifecycle transitions) takes an explicit `now: Instant` parameter rather than reading the system clock internally, so scenarios are exactly reproducible in tests.
- **Math via `nalgebra`, algorithms hand-written.** Linear algebra primitives (matrix ops, inversion) are not reinvented — but the filtering, association, and fusion logic itself is, since that's the actual point of the project.

## Non-goals (for now)

- No mixed ground+air tracking domain.
- No UI/live map.
- No interceptor assignment logic.
- No standards-interoperability layer (e.g. NATO MIP/STANAG export) — plausible future extension once the core is solid, not a current concern.

---

## Tech stack

- Rust
- [`nalgebra`](https://nalgebra.org/) — linear algebra (vectors, matrices, decompositions)
- [`rand`](https://docs.rs/rand/) — seeded simulated sensor noise for deterministic tests

---

## Status log

- ✅ Single-object, single-sensor, 2D constant-velocity Kalman filter — predict/update loop validated by a convergence test against a simulated noisy trajectory.
- 🔜 Extend state to 3D (add altitude/`vz`) — mechanical extension of the same filter.
- 🔜 Multi-object association (step 2 above).