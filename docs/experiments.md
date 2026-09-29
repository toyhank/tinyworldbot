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

## v0.2 pick-and-place notes

The pick-and-place branch started from a deliberately simple question: can the
same cheap simulated arm discover usable grasps without teleoperation?

### Random grasp discovery

Randomized grasp parameters included XY offset, descend height, wrist roll and
gripper target. A lift test automatically labeled each attempt. In an early
95-attempt run, 24 attempts lifted the cube (~25%). A later 140-attempt
collection with broader scene variation produced 21 positives (15%).

This was enough to show that positive examples are not vanishingly rare if the
search is constrained to a small neighborhood around a visually localized
object.

### Lift is not stable transport

A major failure mode was immediately visible: a grasp can lift successfully and
still drop during a lateral move. Adding a carry test created a much stricter
"stable grasp" label. Only a small fraction of random grasps survived that
test.

The current release therefore ships a bank of six primitives that were found by
autonomous exploration and survived lift/carry screening.

### Same-pose wrist verification

A runtime verifier was added so the controller does not need simulator object
height to decide whether to continue.

The robot records an empty-gripper wrist image at a fixed inspection command,
tries a grasp, returns to the same command, and measures RGB residual around the
grasp site.

Diagnostic batches showed that this cue is useful but imperfect. The verifier
can still false-positive on contact-induced pose/image changes, so the release
keeps the threshold and sample sizes visible rather than presenting it as a
solved perception problem.

### Placement correction

Successful carries exposed another bug: the end effector could be centered on
the goal while the object was 1--2 cm off-center inside the gripper.

The v0.2 placement controller compensates for the pre-contact visual
object-to-EE offset and uses a slower table-supported release. On successful
carries this often reduced final error from roughly 1--2 cm to sub-centimeter
or low-centimeter error.

### Physics randomization

The most important negative result is robustness.

A grasp that looked excellent in one MuJoCo domain degraded sharply when cube
size/mass, friction, actuator parameters, camera pose and lighting were
randomized. A fixed grasp produced about 25% complete placement success in both
the moderate and harsh stress-test batches used during development.

That result changed the project direction: the interesting behavior is not
"memorize one perfect grasp", but "detect failure, re-localize, retry, and
eventually adapt on hardware".
