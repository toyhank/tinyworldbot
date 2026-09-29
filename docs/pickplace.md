# Pick-and-place v0.2

TinyWorldBot v0.2 extends the original pushing experiment to a small
self-recovering pick-and-place baseline.

![Pick-and-place](../media/pickplace.gif)

## What is actually learned / discovered

The six grasp primitives in `GRASP_BANK` were not hand-authored as "correct"
grasps. They came from autonomous simulation exploration:

1. sample XY offset, descend height, wrist roll, and gripper target,
2. attempt a physics grasp,
3. lift,
4. use a lift / carry test to label the trial,
5. retain a small bank of grasps that survived.

The current release ships that discovered bank so the benchmark is quick to
reproduce. It does **not** claim that the full pick-and-place controller is an
end-to-end learned policy.

## Runtime pipeline

```text
cached empty-workspace RGB
        |
        v
color-agnostic global object localization
        |
        v
candidate grasp primitive
        |
        v
lift
        |
        v
same-pose wrist-image verification
        |
    fail+release
        |--------------------+
        |                    |
        v                    |
return home                  |
re-detect object             |
try next candidate <---------+
        |
      pass
        |
        v
transport
        |
        v
place EE = goal - estimated object-in-gripper offset
        |
        v
table-support settle
        |
        v
slow gripper unload
```

Runtime candidate selection does not read MuJoCo object XY or object height.
Simulator truth is isolated to final benchmark scoring.

## Self-recovery

A failed attempt is not simply counted and discarded. The controller opens the
gripper, returns to the same home pose used by the cached global background,
re-detects the object, and tries the next self-discovered primitive.

That makes the retry loop structurally similar to something a real SO-101 can
do with a fixed overhead camera and a wrist camera.

## Same-pose wrist verifier

The verifier is intentionally simple.

Before trying a candidate, the robot visits a fixed inspection pose with the
same gripper command that will be used after grasping and records an
empty-gripper wrist image.

After lifting, it returns to the same actuator command and compares a circular
ROI around the grasp site.

A held object changes that ROI. The current baseline uses the 90th percentile
RGB residual as the score.

This is not a semantic detector and does not depend on a hard-coded object
color.

## Placement correction

The original placement controller moved the end effector directly to the goal.
That is wrong whenever the object is off-center inside the gripper.

v0.2 instead estimates the object offset before contact:

```text
object_offset = visual_object_xy - EE_xy
place_EE_xy   = goal_xy - object_offset
```

The release sequence also changed from a short open command to:

1. descend closer to table support,
2. wait for the contact to settle,
3. open over a longer trajectory,
4. wait again before retreating.

On successful carries this substantially reduces final placement error.

## Current benchmark

The small benchmark should be read as an engineering regression test, not a
statistical robotics result.

### Fixed-physics, held-out appearance

The validation setup randomizes object XY and goal XY and uses a held-out
mustard object appearance.

A self-recovering validation batch with threshold fixed beforehand produced:

```text
9 / 16 complete pick-and-place successes = 56.2%
```

A shorter 8-episode subset produced 6 / 8, which is why small samples should
not be over-interpreted.

### Physics randomization

When object size/mass, friction, actuator dynamics, camera pose and lighting
are randomized, the old fixed "champion" grasp falls sharply:

```text
moderate domain: 6 / 24 place successes = 25%
harsh domain:    4 / 16 place successes = 25%
```

That is an important negative result: the original 85% fixed-domain result was
partly exploiting the simulator domain.

The repo therefore does not present "85% sim success" as a sim-to-real claim.

## Reproduce

Run the pick-and-place CLI:

```bash
python -m tinyworldbot.pickplace --trials 8 --seed 13579
```

or, after installation:

```bash
tinyworldbot-pickplace --trials 8 --seed 13579
```

The benchmark uses a fresh simulator/controller per episode so one-time camera
calibration remains reproducible.

The exact 16-episode record quoted above is stored in
[`benchmarks/pickplace_v0.2.json`](../benchmarks/pickplace_v0.2.json).
The physics-randomization stress-test summary is stored in
[`benchmarks/domain_randomization_v0.2.json`](../benchmarks/domain_randomization_v0.2.json).

## Remaining sim-to-real gaps

The controller still assumes:

- a fixed table,
- one-time global camera calibration,
- a cached empty-workspace image,
- known target XY,
- a simple cube-like object,
- a simulated camera mount and camera model,
- grasp primitives discovered in MuJoCo.

The biggest current gap is contact physics. A grasp that is stable for a
MuJoCo friction/actuator setting may not be stable on hardware.

The next useful milestone is not a larger neural network. It is a real SO-101
run with:

- the same cached-background global detector,
- the same side-offset wrist camera,
- the same same-pose wrist verification,
- real retry / re-detection,
- logged failures for friction, backlash, calibration and release.
