# RS-RL

RS-RL is planned as a generic reinforcement-learning training studio for configuring, running, observing, and comparing agent experiments.

The eventual application will combine a Python-based RL and environment layer with Rust performance-critical components and a modern Tauri frontend. It will support GPU or CPU training, live metrics such as reward, loss, throughput, and Elo, configurable dashboards, checkpoint management, environment adapters, and visual replays of agent behaviour.

The project is being developed incrementally. The first milestone is a PPO trainer for a Gymnasium environment with live metrics, checkpointing, and episode replay. Later milestones will add Rust acceleration, custom dashboard layouts, evaluation ladders, Elo tracking, multi-agent environments, and support for a wider range of training systems.

See the [project guide](PROJECT_GUIDE.md) for the architecture, development phases, and implementation recommendations.