# TinyWorldBot

**Can an SO-101 learn useful contact dynamics without teleoperation demonstrations?**

TinyWorldBot is a small MuJoCo research playground for autonomous interaction,
a learned world model, and MPC on an SO-101 arm.

![TinyWorldBot demo](media/demo.gif)

Reference run:

```text
1,600 autonomous interactions
0 teleoperation demonstrations
held-out object appearance

12.81 cm  ->  3.92 cm
27 control steps
success threshold: < 4 cm
```

## Why this exists

A lot of low-cost robot workflows start with:

```text
teleoperate -> record demonstrations -> train policy -> replay
```

TinyWorldBot tests a different, much smaller idea:

```text
explore -> learn local dynamics -> imagine short futures -> act
```

It is not a foundation model and it is not a claim that MPC replaces imitation
learning. The goal is a reproducible bench for studying how far a tiny learned
dynamics model can go on contact manipulation.

## Current pipeline

```text
one-shot overhead RGB
        |
        v
color-agnostic current-vs-empty object bootstrap
        |
        v
geometric setup / choose contact side
        |
        v
side-offset wrist camera
        |
        v
appearance tracking during contact
        |
        v
history-aware learned dynamics
        |
        v
goal-aligned residual CEM/MPC
```

The global camera is used once before the arm moves into the workspace. That is
intentional: before contact, the object should not move, so continuously
tracking it from the overhead view only creates extra opportunities for the arm
to occlude or confuse the tracker.

Near contact, perception hands off to a wrist camera. The simulated wrist
camera is moved to a small side-offset mount because the stock model's camera
was occluded in part of the contact workspace.

## What the controller does *not* get

During data collection, training, and action selection:

- no MuJoCo object XY,
- no segmentation labels,
- no teleoperation demonstrations,
- no hard-coded object color threshold.

MuJoCo object XY is read only for the final evaluation distance.

There are still important priors:

- a fixed tabletop,
- a one-time camera calibration,
- an empty-scene reference for global bootstrap,
- a known goal XY,
- simple box-like objects in the current benchmark,
- a geometric high-level setup controller.

Those are features of the baseline, not hidden details.

## Quick start

Python 3.10+ is recommended. A CUDA GPU makes CEM/model training faster, but CPU
execution is supported.

```bash
git clone https://github.com/toyhank/tinyworldbot.git
cd tinyworldbot

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e .
python -m tinyworldbot.demo --samples 1600 --epochs 35
```

A shorter smoke run:

```bash
python -m tinyworldbot.demo --samples 300 --epochs 3 --device cpu
```

The demo writes `outputs/demo.gif`.

## Method

The learned state is:

```text
[EE_x, EE_y, object_x, object_y, joint_1 ... joint_5]
```

The object position comes from RGB perception, not simulator object state. The
world model receives the current state, two state deltas, the current action,
and two previous actions. A learned contact gate modulates predicted object
motion.

The planner does not search arbitrary XY commands. It searches a compact
goal-aligned action basis:

```text
forward push magnitude + lateral correction
```

This was substantially more stable than unconstrained XY CEM in the same
simulation.

## Results and negative results

See [docs/experiments.md](docs/experiments.md). The failed variants are useful
because they show where the current system is brittle: long-horizon model
exploitation and visual object permanence under contact.

## Repository layout

```text
tinyworldbot/
  env.py          # SO-101 MuJoCo environment + IK
  vision.py       # global bootstrap + wrist tracking
  world_model.py  # history-aware dynamics model
  planner.py      # setup controller + residual CEM/MPC
  demo.py         # autonomous data collection, training, evaluation
  assets/so101/   # MuJoCo model and meshes
tests/
docs/
media/
```

## What would make this more interesting

The next milestones are intentionally harder than polishing the current
simulation number:

- real SO-101 transfer,
- real camera calibration and wrist mount,
- multiple object shapes/materials,
- moving-camera mask tracking instead of appearance tracking,
- camera and lighting perturbations,
- comparison against a hand-coded push baseline and behavior cloning,
- cross-embodiment transfer.

## Reproducibility note

The headline number above is one deterministic reference run from the current
standalone repository (seed 113, 1,600 interactions, 35 epochs). It is **not** presented as a statistically established
success rate. Multi-seed evaluation is the next benchmark to add.

## Third-party assets

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The bundled SO-101 model
assets retain their Apache-2.0 licensing and attribution.
