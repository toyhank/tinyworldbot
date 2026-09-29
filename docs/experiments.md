# Experiment notes

TinyWorldBot started as a deliberately small question: can a cheap arm learn
enough local contact dynamics from its own interaction data to push an object
to a goal without teleoperation demonstrations?

## What failed

The failures are part of the project, not hidden tuning history.

| Variant | Result | Main failure |
| --- | --- | --- |
| Plain learned MPC | unstable | optimizer exploited model error |
| Contact-gated MPC | improved | wrong-side contact remained hard |
| History model only | unstable | perception/contact topology still mattered |
| Generic overhead KLT/CamShift | about 6 cm | tracker drifted onto the gripper |
| DINOv2 patch/template tracking | too noisy | semantic features were not precise enough for millimeter-scale contact |

## What worked

A hierarchical controller worked better than asking a single learned rollout to
solve free-space repositioning and contact at the same time:

1. one-shot global RGB localization before the arm enters the scene,
2. geometric setup to approach from the correct side,
3. handoff to a side-offset wrist camera before contact,
4. history-aware learned local dynamics,
5. goal-aligned residual CEM/MPC during pushing.

The current reference run uses 1,600 autonomous transitions and 35 training
epochs, then evaluates from a deliberately wrong-side initialization.

Observed reference run:

- start distance: 12.81 cm
- final distance: 3.92 cm
- 27 control steps
- success threshold: < 4 cm

## Important ablation

When object isolation was made artificially perfect in simulation, the same
world-model/MPC stack also reached the success threshold. That experiment was
useful diagnostically but is intentionally not the main method because it used
a simulation-only counterfactual render.

The current main demo instead uses a practical camera split: global vision for
initial localization, wrist vision for contact.
