# RS-RL Project Guide

## 1. Project vision

RS-RL is intended to become a generic reinforcement-learning trainer with a modern, highly configurable interface. It should make it possible to configure experiments, train agents efficiently on a GPU or CPU, monitor training in real time, compare checkpoints, and inspect visual replays of agent behaviour.

The project should support a wide range of training environments without coupling the frontend to a particular RL library or algorithm. The key design principle is to keep the trainer, environment, data, and presentation layers separate.

The eventual user experience should include:

- Live training metrics such as reward, loss, entropy, throughput, Elo, and evaluation results.
- Customisable dashboards where users choose which metrics and visualisations are shown.
- Checkpoint management, experiment comparison, and resumable training.
- Environment adapters that can expose observations, actions, metadata, and rendered replays.
- GPU training by default where supported, with configurable CPU execution.
- A fast native layer for performance-sensitive data handling.
- Support for adding new algorithms, environments, metrics, and visualisations without rewriting the application.

This guide is a practical development path. It intentionally starts with a small, working system before expanding toward a general-purpose platform.

## 2. Recommended architecture

Use each language where it provides the most value:

### Python

Python should initially own:

- Gymnasium and PettingZoo-compatible environments.
- RL algorithms and training logic.
- PyTorch or JAX integration.
- Environment adapters.
- Experiment configuration.
- High-level data processing.

Python has the strongest ecosystem for RL and deep learning, so rewriting algorithms in Rust at the beginning would add complexity without a guaranteed performance benefit.

### Rust

Rust should own components that are CPU-bound, performance-sensitive, or long-running:

- Rollout and trajectory buffers.
- Experience replay storage.
- Vectorised environment communication where profiling justifies it.
- Metrics aggregation and downsampling.
- Image or frame encoding.
- Native application coordination and IPC.
- Storage and data movement utilities.

Add Rust incrementally. Profile first, then move bottlenecks rather than assuming that every Python component needs to be rewritten.

### Frontend

The recommended desktop stack is:

```text
Tauri + React + TypeScript
```

Tauri gives the application a native Rust layer while allowing a modern web frontend. The frontend should communicate with the trainer through typed events and commands rather than directly depending on Gymnasium, PyTorch, or internal Python classes.

For an initial web-only prototype, React or Svelte with a Python WebSocket/API layer is also valid. Tauri becomes more attractive when native packaging, filesystem access, and Rust integration are important.

## 3. Python and Rust integration

Use [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/) to expose Rust code as a Python extension:

```text
rs-rl/
├── python/
│   ├── rs_rl/
│   └── pyproject.toml
├── rust/
│   ├── rust_core/
│   │   └── src/
│   └── Cargo.toml
└── README.md
```

Python can then import the compiled module like any other dependency:

```python
import rs_rl_core
```

A typical data flow is:

```text
Environment
    -> NumPy or Torch tensors
    -> Rust rollout buffer
    -> Python batching layer
    -> PyTorch GPU training
    -> metrics and checkpoint events
    -> Rust or application event layer
    -> frontend
```

Keep GPU tensor operations in PyTorch initially. Rust is most useful for CPU-side storage, batching, encoding, and communication. Avoid repeatedly copying large tensors between Python, Rust, and the frontend.

## 4. Minimum viable product

Do not begin by supporting every algorithm and environment. The first milestone should be:

> Train PPO in one Gymnasium environment, stream live metrics, show a configurable reward chart, save checkpoints, and replay an episode.

Use the following initial scope:

- One environment API: Gymnasium.
- One algorithm: PPO.
- One ML framework: PyTorch.
- One environment family: a simple vectorisable environment with rendering.
- One execution mode: CUDA with CPU fallback.
- One frontend: Tauri with React and TypeScript.
- One event format: typed serialisable training events.

Suitable first environments include CartPole, LunarLander, or another small environment that can be rendered reliably. A simple environment makes it easier to validate the trainer and frontend independently.

## 5. Core components

### Trainer

The trainer should expose lifecycle operations without exposing implementation details to the frontend:

```python
class Trainer:
    def start(self) -> None: ...
    def pause(self) -> None: ...
    def stop(self) -> None: ...
    def step(self) -> "TrainingResult": ...
```

The concrete API can evolve, but it should support:

- Start, pause, resume, and stop.
- Single-step or short-step execution for debugging.
- Configurable device selection.
- Periodic evaluation.
- Checkpoint creation and restoration.
- Graceful error reporting.

### Environment adapter

The frontend should not know about Gymnasium APIs. Introduce an adapter boundary that can later support other frameworks:

```python
class EnvironmentAdapter:
    def reset(self): ...
    def step(self, actions): ...
    def render(self) -> bytes: ...
```

The real interface should use precise project types and include environment metadata, terminated/truncated state, action information, and optional replay data.

### Event stream

Define a stable event stream as the main boundary between the trainer and frontend:

```text
Trainer event
├── metric
├── episode
├── replay
├── checkpoint
├── trainer status
└── evaluation match
```

The frontend should consume events, not inspect trainer internals. This lets the same backend support multiple frontend views and allows new metrics to be added without hardcoding every field into the UI.

JSON is appropriate for the first prototype. As event volume increases, consider MessagePack, CBOR, or Protocol Buffers. Whichever format is chosen, define schemas and version them.

Example metric event:

```json
{
  "run_id": "ppo-cartpole-001",
  "step": 100000,
  "wall_time": 1730000000.0,
  "values": {
    "reward_mean": 152.4,
    "policy_loss": -0.021,
    "value_loss": 0.084
  }
}
```

## 6. Metrics and dashboard design

The initial metric set should include:

```text
step
episode
reward_mean
reward_std
episode_length_mean
policy_loss
value_loss
entropy
fps
learning_rate
elapsed_seconds
GPU memory
GPU utilisation
```

Metrics should have names, types, units where applicable, timestamps or training steps, and a source. Environment-specific metrics should be allowed without requiring a frontend release.

Losses and performance metrics should be kept conceptually separate:

- Training losses describe optimisation behaviour.
- Reward and episode length describe environment performance.
- Elo, win rate, and evaluation scores describe competition or evaluation.
- Hardware metrics describe resource usage.

Use a declarative dashboard layout. A widget could contain fields similar to:

```typescript
interface DashboardWidget {
  id: string;
  title: string;
  metric?: string;
  kind: "line" | "scalar" | "heatmap" | "video" | "table";
  size: { width: number; height: number };
  visible: boolean;
}
```

Potential widget types include:

- Scalar cards for current values.
- Line charts for reward and loss.
- Distributions for episode returns.
- Tables for evaluation matches.
- Video or frame viewers for replays.
- Heatmaps for action probabilities or state coverage.

Keep rendering work off the trainer thread. Aggregate or downsample high-frequency metrics before sending them to the UI so a fast trainer does not overwhelm the frontend.

## 7. Replay and environment data

Replay support should be optional and configurable because capturing and encoding frames can be expensive.

A replay record may contain:

```text
episode metadata
timestep
observation metadata
action
reward
terminated/truncated state
optional rendered frame
```

A practical first implementation is:

1. Select episodes for recording at a configurable interval.
2. Capture frames from the environment.
3. Encode frames as PNG or WebP.
4. Store frames in the run directory or a managed data store.
5. Send the frontend replay metadata and frame references.
6. Allow pause, seeking, speed control, and episode selection.

Do not send every raw frame over IPC by default. Sample episodes and use compressed, chunked data. Later, add environment-specific replay providers for environments where a complete visual frame is not the best representation.

## 8. Checkpoints, runs, and persistence

Checkpointing should be implemented early. A checkpoint should include:

- Model weights.
- Optimiser state.
- Random-number-generator state.
- Environment version and configuration.
- Algorithm hyperparameters.
- Training and episode counters.
- Relevant metrics and run metadata.
- References to replay data where applicable.

Write checkpoints atomically so an interrupted write cannot appear to be a valid checkpoint. The application should support starting a run, pausing it, resuming it later, and comparing multiple checkpoints.

Use a simple run directory structure first. Add SQLite or another embedded metadata database when querying many runs, metrics, and evaluations becomes useful.

## 9. Elo and evaluation

Elo is an evaluation system, not a replacement for training loss or reward. Keep evaluation separate from the training loop where possible.

Evaluate checkpoints against:

- A previous checkpoint.
- A moving copy of the current agent.
- Random or heuristic baselines.
- Other agents.
- Human or external opponents where appropriate.

Store each evaluation match as structured data. Calculate and display Elo, win rate, draw rate, and confidence information from the evaluation records rather than calculating everything only in the UI.

The meaning of any metric called “expected loss” must be defined explicitly. It could refer to value prediction error, forecasted loss, uncertainty, or another quantity. Give it a precise name and documentation once its mathematical definition is chosen.

## 10. Performance strategy

Optimise in this order:

1. Environment throughput and vectorisation.
2. GPU batch size and tensor placement.
3. Unnecessary CPU/GPU transfers.
4. Replay and rollout buffer allocation.
5. Python loops in environment stepping.
6. Metrics aggregation and IPC volume.
7. Frame encoding and disk writes.

Use profiling to decide what belongs in Rust. Keep tensors on the GPU through the training step where practical, and transfer only the data needed for metrics, persistence, or replay.

Avoid introducing distributed training, microservices, or a custom environment engine until a measured bottleneck requires them.

## 11. Suggested development phases

### Phase 1: working training loop

- Create a Python package.
- Add PPO and one Gymnasium environment.
- Support CPU and CUDA device selection.
- Add basic configuration.
- Run a complete training session from the command line.

### Phase 2: runs and metrics

- Add run IDs and metadata.
- Add checkpoint saving and restoration.
- Define metric and status events.
- Stream or persist metrics.
- Add evaluation episodes.

### Phase 3: frontend prototype

- Create the Tauri application.
- Show trainer status.
- Display a live reward chart.
- Display selected scalar metrics.
- Add start, pause, resume, and stop controls.

### Phase 4: replay

- Add an environment rendering adapter.
- Record selected episodes.
- Store compressed frames.
- Add playback, seeking, and episode selection.

### Phase 5: Rust performance layer

- Profile the system.
- Move the rollout or replay buffer to Rust if it is a bottleneck.
- Expose the component with PyO3 and maturin.
- Add typed boundaries and benchmarks.

### Phase 6: customisation and evaluation

- Add configurable dashboard widgets.
- Add run comparison.
- Add evaluation match records.
- Add Elo and checkpoint ladders.
- Add environment-specific plugin metrics.

### Phase 7: wider environment support

- Add adapters for additional environment APIs.
- Add multi-agent support through PettingZoo or a compatible abstraction.
- Add more replay formats.
- Improve packaging, documentation, and failure recovery.

## 12. Design rules to preserve

- Keep the frontend independent of Python and specific RL libraries.
- Keep environment-specific behaviour behind adapters.
- Make events and persisted records versioned and serialisable.
- Treat replay capture as optional.
- Treat evaluation metrics separately from optimisation metrics.
- Profile before moving code to Rust.
- Prefer one reliable process over many services initially.
- Make failures visible to the user rather than silently falling back.
- Test the trainer, event stream, checkpoint restoration, and frontend boundary independently.

## 13. Recommended initial technology stack

```text
Python       PyTorch, Gymnasium, NumPy
Rust         Cargo, PyO3, maturin
Desktop UI   Tauri, React, TypeScript
Charts       ECharts, Plotly, or another suitable chart library
Storage      Run directories first, SQLite when run querying grows
Transport    JSON initially, then a typed binary format if required
```

The first implementation should favour a complete, observable training workflow over maximum theoretical performance. Once the workflow is correct and measurable, Rust can be introduced where it produces a meaningful improvement.
