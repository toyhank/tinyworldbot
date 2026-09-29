from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from .env import SO101PushEnv
from .vision import (
    GlobalBootstrapDetector,
    HighPrecisionGlobalDetector,
    configure_offset_wrist_camera,
    poly_features,
)


# These six primitives were found by autonomous random exploration followed by
# lift/carry self-supervision in the development experiments. They are a small
# behavior bank, not hand-authored "correct" grasps.
GRASP_BANK = np.asarray(
    [
        [0.0040, -0.0106, 0.0253, 1.0403, 0.0274],
        [0.0030, -0.0097, 0.0202, 0.9650, -0.0537],
        [0.0051, -0.0113, 0.0240, 0.9928, -0.0146],
        [0.0048, -0.0119, 0.0262, 1.2653, -0.0580],
        [0.0039, -0.0141, 0.0240, 1.4185, -0.0139],
        [-0.0110, 0.0020, 0.0282, 0.7022, 0.1090],
    ],
    dtype=np.float32,
)


@dataclass
class EpisodeResult:
    success: bool
    error_cm: float
    attempts: int
    candidate: int | None


class SamePoseWristVerifier:
    """Detect a held object by comparing the wrist view at the same command.

    Before the grasp, the robot visits a fixed inspection pose with the
    candidate gripper command and records an empty-gripper reference. After
    lift, it returns to the same actuator command. A large image residual near
    the grasp site is evidence that something is still in the gripper.

    This deliberately avoids simulator object coordinates.
    """

    def __init__(self, env: SO101PushEnv, width=320, height=240) -> None:
        self.env = env
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(
            env.model, height=height, width=width
        )
        self.camera_id = env.model.camera("wrist_camera").id
        fovy = np.deg2rad(float(env.model.cam_fovy[self.camera_id]))
        self.focal = 0.5 * height / np.tan(0.5 * fovy)

    def close(self) -> None:
        self.renderer.close()

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.data, camera="wrist_camera")
        return self.renderer.render().copy()

    def _site_pixel(self) -> np.ndarray:
        camera_pos = self.env.data.cam_xpos[self.camera_id].copy()
        camera_rot = self.env.data.cam_xmat[
            self.camera_id
        ].reshape(3, 3).copy()
        site = self.env.data.site_xpos[self.env._site_id].copy()
        q = camera_rot.T @ (site - camera_pos)
        depth = -float(q[2])
        return np.asarray(
            [
                self.width / 2 + self.focal * q[0] / depth,
                self.height / 2 - self.focal * q[1] / depth,
            ],
            dtype=np.float32,
        )

    def p90_residual(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        radius: float = 48.0,
    ) -> float:
        uv = self._site_pixel()
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        roi = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= radius ** 2
        diff = np.linalg.norm(
            current.astype(np.float32) - reference.astype(np.float32),
            axis=2,
        )
        return float(np.percentile(diff[roi], 90))


class CachedObjectDetector:
    """Color-agnostic global detector using a cached empty-workspace image."""

    def __init__(
        self,
        calibrated_detector: GlobalBootstrapDetector,
        background: np.ndarray,
    ) -> None:
        self.detector = calibrated_detector
        self.background = background.copy()

    def locate(self) -> np.ndarray:
        current = self.detector.render()
        diff = np.linalg.norm(
            current.astype(np.float32) - self.background.astype(np.float32),
            axis=2,
        )
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        nonzero = diff[diff > 2.0]
        if len(nonzero) < 8:
            raise RuntimeError("object is not visible in the global view")

        threshold = max(10.0, float(np.percentile(nonzero, 65)))
        mask = (diff >= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        candidates: list[tuple[float, int]] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 8 <= area <= 2200 and w <= 70 and h <= 70:
                score = float(diff[labels == i].sum())
                candidates.append((score, i))
        if not candidates:
            raise RuntimeError("no compact object component was found")

        _, idx = max(candidates)
        x, y, w, h, _ = stats[idx]
        uv = np.asarray([x + w / 2, y + h / 2], dtype=np.float32)
        return (
            poly_features(uv) @ calibrated_detector_world_map(self.detector)
        ).astype(np.float32)


def calibrated_detector_world_map(
    detector: GlobalBootstrapDetector,
) -> np.ndarray:
    # GlobalBootstrapDetector calls this matrix world_w. Keeping access in one
    # helper makes the dependency explicit and easy to replace with a camera
    # model later.
    if detector.world_w is None:
        raise RuntimeError("global detector is not calibrated")
    return detector.world_w


class PickPlaceController:
    """Self-recovering pick-and-place baseline for the simulated SO-101."""

    inspection = np.asarray([0.295, 0.075, 0.095], dtype=np.float64)
    verifier_threshold = 75.0

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.env = SO101PushEnv(seed=seed)
        configure_offset_wrist_camera(self.env)
        self.global_detector = HighPrecisionGlobalDetector(self.env)
        self.wrist = SamePoseWristVerifier(self.env)
        self.home_q = self.env.joint_positions.astype(float).copy()

    def close(self) -> None:
        self.wrist.close()
        self.global_detector.close()

    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.env.model, self.env.data)

    def _move_xyz(
        self,
        target: np.ndarray | list[float],
        rotation: np.ndarray,
        max_step: float = 0.005,
    ) -> bool:
        start = self.env.data.site_xpos[self.env._site_id].copy()
        target = np.asarray(target, dtype=float)
        segments = max(
            4, int(np.ceil(np.linalg.norm(target - start) / max_step))
        )
        for fraction in np.linspace(1 / segments, 1.0, segments):
            waypoint = start + fraction * (target - start)
            ok, joints, _ = self.env.solve_ik(waypoint, rotation)
            if not ok:
                return False
            joints[-1] = self.env.joint_positions[-1]
            self.env._joint_trajectory(joints, steps=1)
        return True

    def _command_ctrl(self, target: np.ndarray, steps: int = 10) -> None:
        start = self.env.data.ctrl.copy()
        for fraction in np.linspace(1 / steps, 1.0, steps):
            self.env.data.ctrl[:] = start + fraction * (target - start)
            for _ in range(17):
                mujoco.mj_step(self.env.model, self.env.data)

    def _capture_background(self) -> np.ndarray:
        """Simulation helper for the real-world 'empty table' calibration.

        This is called before runtime object detection. On hardware, save one
        empty-workspace image instead of moving an object in code.
        """
        address = self.env._cube_joint_addresses["red_cube"]
        qpos = self.env.data.qpos[address : address + 7].copy()
        qvel = self.env.data.qvel.copy()
        self.env.write_cube_pose(
            "red_cube",
            np.asarray([0.60, 0.60, self.env.cube_half_size]),
        )
        mujoco.mj_forward(self.env.model, self.env.data)
        image = self.global_detector.render()
        self.env.data.qpos[address : address + 7] = qpos
        self.env.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        return image

    def reset_episode(self, seed: int) -> tuple[CachedObjectDetector, np.ndarray]:
        self.env.reset(seed)
        self.env.hide_distractors()
        rng = np.random.default_rng(seed)

        # Environment setup only. These coordinates are never passed to the
        # controller after the scene has been created.
        object_xy = np.asarray(
            [rng.uniform(0.195, 0.270), rng.uniform(-0.075, 0.025)],
            dtype=np.float32,
        )
        self.env.write_cube_pose(
            "red_cube",
            np.r_[object_xy, self.env.cube_half_size],
        )
        # Held-out appearance relative to the training palette.
        self.env.model.geom_rgba[self.env._cube_geom_ids["red_cube"]] = (
            0.56,
            0.51,
            0.17,
            1.0,
        )
        mujoco.mj_forward(self.env.model, self.env.data)

        self.home_q = self.env.joint_positions.astype(float).copy()
        background = self._capture_background()
        detector = CachedObjectDetector(self.global_detector, background)

        goal_rng = np.random.default_rng(seed + 17)
        goal = np.asarray(
            [
                goal_rng.uniform(0.280, 0.315),
                goal_rng.uniform(0.075, 0.120),
            ],
            dtype=np.float32,
        )
        return detector, goal

    def _go_home(self) -> None:
        # Clear the table before returning to the exact background pose.
        try:
            site = self.env.data.site_xpos[self.env._site_id].copy()
            rotation = self.env.data.site_xmat[
                self.env._site_id
            ].reshape(3, 3).copy()
            self._move_xyz(
                [site[0], site[1], 0.115], rotation, max_step=0.006
            )
        except Exception:
            pass
        self.env.set_joints_direct(self.home_q)
        self._settle(80)

    def _probe(
        self,
        object_xy: np.ndarray,
        params: np.ndarray,
    ) -> dict[str, object] | None:
        dx, dy, z, roll, close = map(float, params)
        q = self.env.joint_positions.astype(float)
        q[-1] = 1.55
        q[4] = roll
        self.env.set_joints_direct(q)
        rotation = self.env.data.site_xmat[
            self.env._site_id
        ].reshape(3, 3).copy()

        if not self._move_xyz(self.inspection, rotation):
            return None
        q = self.env.joint_positions.astype(float)
        q[-1] = close
        self.env._joint_trajectory(q, steps=12)
        self._settle(80)
        reference_ctrl = self.env.data.ctrl.copy()
        reference = self.wrist.render()

        q = self.env.joint_positions.astype(float)
        q[-1] = 1.55
        self.env._joint_trajectory(q, steps=8)

        grasp_xy = np.asarray(object_xy, float) + np.asarray([dx, dy])
        if not self._move_xyz([*grasp_xy, 0.095], rotation):
            return None
        if not self._move_xyz([*grasp_xy, z], rotation, max_step=0.004):
            return None

        # Available before contact: visual object center minus robot FK.
        object_offset = (
            np.asarray(object_xy, float) - self.env.ee_xy.copy()
        )

        q = self.env.joint_positions.astype(float)
        q[-1] = close
        self.env._joint_trajectory(q, steps=16)
        self._settle(30)
        if not self._move_xyz([*grasp_xy, 0.100], rotation, 0.0035):
            return None
        if not self._move_xyz(self.inspection, rotation, 0.0035):
            return None

        self._command_ctrl(reference_ctrl, steps=10)
        self._settle(100)
        residual = self.wrist.p90_residual(
            reference, self.wrist.render()
        )
        passed = residual > self.verifier_threshold
        return {
            "passed": passed,
            "p90": residual,
            "rotation": rotation,
            "offset": object_offset,
        }

    def _release_failed_probe(self) -> None:
        q = self.env.joint_positions.astype(float)
        q[-1] = 1.55
        self.env._joint_trajectory(q, steps=36)
        self._settle(180)

    def _place(self, probe: dict[str, object], goal: np.ndarray) -> bool:
        rotation = np.asarray(probe["rotation"], dtype=float)
        offset = np.asarray(probe["offset"], dtype=float)

        # Center the estimated object, not the end effector, over the target.
        target = np.asarray(goal, float) - offset
        correction = np.clip(target - goal, -0.025, 0.025)
        target = np.asarray(goal, float) + correction
        target[0] = np.clip(target[0], *self.env.xlim)
        target[1] = np.clip(target[1], *self.env.ylim)

        midpoint = np.asarray(
            [
                0.5 * (self.inspection[0] + target[0]),
                0.5 * (self.inspection[1] + target[1]),
                0.105,
            ]
        )
        if not self._move_xyz(midpoint, rotation, 0.0030):
            return False
        if not self._move_xyz([*target, 0.105], rotation, 0.0030):
            return False
        if not self._move_xyz([*target, 0.022], rotation, 0.0022):
            return False

        # Let the table take load before gradually unloading the fingers.
        self._settle(100)
        q = self.env.joint_positions.astype(float)
        q[-1] = 1.55
        self.env._joint_trajectory(q, steps=64)
        self._settle(180)
        self._move_xyz([*target, 0.095], rotation, 0.0030)
        self._settle(120)
        return True

    def evaluate_truth(self, goal: np.ndarray) -> tuple[bool, float]:
        """Evaluation only; never used to choose an action."""
        object_xy = self.env.true_object_xy()
        object_z = float(
            self.env.data.xpos[self.env._cube_body_ids["red_cube"], 2]
        )
        error = float(np.linalg.norm(object_xy - goal))
        return bool(object_z < 0.045 and error < 0.050), error

    def run_episode(self, seed: int, max_attempts: int = 6) -> EpisodeResult:
        detector, goal = self.reset_episode(seed)
        self._go_home()
        object_xy = detector.locate()

        for index, params in enumerate(GRASP_BANK[:max_attempts], start=1):
            probe = self._probe(object_xy, params)
            if probe is not None:
                print(
                    f"candidate={index} p90={probe['p90']:.1f} "
                    f"pass={int(bool(probe['passed']))}"
                )
            if probe is not None and bool(probe["passed"]):
                self._place(probe, goal)
                success, error = self.evaluate_truth(goal)
                return EpisodeResult(
                    success=success,
                    error_cm=100 * error,
                    attempts=index,
                    candidate=index,
                )

            self._release_failed_probe()
            self._go_home()
            try:
                object_xy = detector.locate()
            except RuntimeError:
                break

        success, error = self.evaluate_truth(goal)
        return EpisodeResult(
            success=False,
            error_cm=100 * error,
            attempts=min(max_attempts, len(GRASP_BANK)),
            candidate=None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-recovering SO-101 pick-and-place demo."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13579)
    parser.add_argument("--max-attempts", type=int, default=6)
    args = parser.parse_args()

    results: list[EpisodeResult] = []
    for i in range(args.trials):
        seed = args.seed + i * 7919
        print(f"\nEPISODE {i+1:02d}/{args.trials}")
        # Fresh simulator/controller per episode keeps one-time camera
        # calibration independent and makes benchmark episodes reproducible.
        controller = PickPlaceController(seed=seed)
        try:
            result = controller.run_episode(
                seed, max_attempts=args.max_attempts
            )
        finally:
            controller.close()
        results.append(result)
        print(
            f"result success={int(result.success)} "
            f"error={result.error_cm:.2f}cm "
            f"attempts={result.attempts} "
            f"candidate={result.candidate}"
        )

    successes = sum(int(r.success) for r in results)
    errors = np.asarray([r.error_cm for r in results])
    attempts = np.asarray([r.attempts for r in results])
    print(
        f"\nSUMMARY success={successes}/{len(results)}="
        f"{successes/max(len(results),1):.1%} "
        f"median_error={np.median(errors):.2f}cm "
        f"mean_attempts={attempts.mean():.2f}"
    )


if __name__ == "__main__":
    main()
