# DOPA-SYNC methodology and environment contract

## Shared discrete-time contract

The intersection has two aggregate queues: north–south (NS) and east–west (EW). Phase 0 gives NS green and EW red; phase 1 gives EW green and NS red. There is no yellow or all-red interval: a permitted switch is atomic. Since exactly one phase is active at all times, conflicting green signals cannot occur. The decision interval is `step_seconds` (one second by default), minimum green is `minimum_green_steps` intervals, and `green_age` is capped at that minimum. Actions are `KEEP=0` and `SWITCH=1`. Before the minimum is met, only KEEP is legal.

At each step, Poisson arrivals for both queues are sampled (or an explicit arrival pair is supplied to parity tests), then every waiting vehicle accumulates one step of waiting time, then up to `discharge_per_step` vehicles leave from the active green queue. Completed-vehicle count and completed wait are updated only at departure. The state is `(phase, ns_queue_bin, ew_queue_bin, green_age_capped)`. Queue bins use the configured boundaries as `[0,3)`, `[3,7)`, `[7,12)`, and `[12,∞)` under defaults; the first boundary is the required zero sentinel. Integer encoding is

`(((phase * bin_count + ns_bin) * bin_count + ew_bin) * (minimum_green_steps + 1) + green_age_capped)`.

Reward per step is `-(NS queue after service + EW queue after service) - switching_cost * switched`. The default switching cost is zero. Reward therefore measures remaining aggregate queue burden in vehicle-steps per decision interval, not travel time or throughput directly. The metric `waiting_vehicle_seconds` counts the queue present after arrivals and before service times `step_seconds`; `mean_wait_seconds_per_completed_vehicle` averages each completed vehicle's accumulated wait. Queue lengths and arrivals are counts of vehicles.

The Q update uses the greatest next-state Q value among actions legal under the next state's minimum-green constraint. For terminal transitions the continuation is zero. The browser uses the same legal-action restriction to compute the TD diagnostic and never updates the exported Q-table. Discretization loses within-bin queue detail; reward can trade waiting against throughput and phase allocation, and zero switching penalty does not explicitly discourage frequent legal switches.

## Synthetic demand and experimental protocol

Each movement uses an independent Poisson count per decision interval with rates in `config.json`. This is a synthetic process and is not calibrated against measured traffic. Training episodes use seeds `seed + episode_index`; held-out paired evaluation uses the configured `seed_start` range. For a given evaluation seed, the RL and fixed-time runs start separate environments with the same seed, so their arrival sequences match because both sample both movements once per step. The fixed controller changes after its configured number of steps, subject to the environment's minimum-green restriction. No model selection is performed on held-out evaluation results.

The reported time-averaged queue metric is aggregate queue occupancy divided by two movements and episode duration. Return is the sum of per-step rewards. Throughput is the number of vehicles discharged. Switches count phase changes. Evaluation summary variability is sample standard deviation (`ddof=1`) across held-out scenarios; no statistical significance test is performed. A stable curve is only a diagnostic, not evidence of global optimality.

## Scientific interpretation

Temporal-difference error is computed as `δ = r + γ max_a Q(s′,a) − Q(s,a)` for a nonterminal step and `δ = r − Q(s,a)` at termination. In this project it is a computational reward prediction error inspired by dopamine research. It is not a biochemical simulation, dopamine concentration, neural firing measurement, or complete model of the brain. The traffic experiment tests a tabular controller in a synthetic environment; its performance cannot validate biological dopamine mechanisms.

## Python/browser consistency

The policy JSON declares the schema, Q-table, action IDs, state encoding, queue edges, environment parameters, training parameters, and SHA-256 over canonical compact JSON with sorted object keys and integral floats normalized to integers. Python validates the checksum and Q-values after export; JavaScript validates checksum, dimensions, action mapping, and state bounds before use. Both languages implement the same queue bin thresholds, encoding formula, action legality, order of arrivals/waiting/service, reward, terminal handling, and greedy tie behavior (lowest legal action ID). Cross-language tests provide explicit deterministic arrivals; the browser's seeded arrival generator is a reproducible JavaScript stream but is not intended to reproduce NumPy's RNG bit-for-bit.
