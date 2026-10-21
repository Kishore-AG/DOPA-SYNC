# DOPA-SYNC Experimental Report

## Abstract

Pending real training and held-out evaluation run.

## Introduction

Describe the adaptive signal-control question and scope. Avoid claims of biological validation or novelty.

## Problem statement

State the two-phase synthetic queue-control problem, decision interval, and demand assumptions.

## Objectives

1. Train tabular Q-learning through environment interaction.
2. Compare its greedy policy with a fixed-time controller on paired held-out seeds.
3. Inspect its temporal-difference reward prediction errors as a computational analogy.

## Scientific background

Separate established reinforcement-learning theory from dopamine-related findings and project design choices. Cite verified sources listed in the README.

## Methodology

Document the environment contract in `docs/methodology.md`, training parameters, independent seeds, and artifact validation.

## Environment and state/action/reward design

Report the actual config used. Include phase encoding, queue bins, legal actions, minimum-green rule, reward units, and synthetic arrival rates.

## Training configuration

Pending run. Attach the generated `artifacts/training_metadata.json`.

## Experimental protocol

Use identical held-out arrival seeds per controller. Report the number of scenarios and note that no significance test is performed.

## Results

**Pending actual runs. Do not fill with expected or illustrative values.** Insert the generated `results/experiment_summary.json`, CSV tables, and plots after executing the notebook.

## Discussion

Discuss measured comparisons only, including scenario variability and the limitations of the baseline and reward.

## Limitations

Synthetic uncalibrated traffic; one simplified intersection; aggregate opposing movements; discretized queues; tabular state/action space; no turns, pedestrians, emergency vehicles, or inter-intersection coordination; computational prediction-error model rather than dopamine measurement; limited seeds and no inferential significance analysis.

## Conclusion

Summarize only what the executed experiments support. A traffic simulation does not validate biological dopamine mechanisms.

## References

Use the verified references in the project README and include DOI links where applicable.
