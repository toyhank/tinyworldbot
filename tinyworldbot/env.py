from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "so101" / "scene.xml"


@dataclass(frozen=True)
class JointSpec:
    names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    lower: tuple[float, ...] = (
        -1.919862, -1.745329, -1.69, -1.658063, -2.743847, -0.174533,
    )
    upper: tuple[float, ...] = (
        1.919862, 1.745329, 1.69, 1.658063, 2.841206, 1.745329,
    )
    home: tuple[float, ...] = (0.0, -0.75, 1.20, 0.85, 0.0, 1.45)

    def lower_array(self) -> np.ndarray:
        return np.asarray(self.lower, dtype=np.float32)

    def upper_array(self) -> np.ndarray:
        return np.asarray(self.upper, dtype=np.float32)

    def home_array(self) -> np.ndarray:
        return np.asarray(self.home, dtype=np.float64)


JOINTS = JointSpec()
CUBE_NAMES = ("red_cube", "green_cube", "yellow_cube", "purple_cube")


class SO101PushEnv:
    """Small SO-101 MuJoCo pushing environment.

    The policy acts in end-effector XY deltas. Each action is converted to
    joint targets with damped least-squares IK and executed by MuJoCo position
    actuators.

    true_object_xy() exists for evaluation only. The demo does not feed it
    into training or action selection.
    """

    cube_half_size = 0.018
    push_z = 0.026
    xlim = (0.13, 0.33)
    ylim = (-0.14, 0.14)
    action_limit = 0.018

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self._ik_work = mujoco.MjData(self.model)

        self._joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINTS.names
        }
        self._joint_qpos = {
            name: int(self.model.jnt_qposadr[jid])
            for name, jid in self._joint_ids.items()
        }
        self._actuator_ids = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in JOINTS.names
        }

        self._cube_joint_addresses: dict[str, int] = {}
        self._cube_dof_addresses: dict[str, int] = {}
        self._cube_geom_ids: dict[str, int] = {}
        self._cube_body_ids: dict[str, int] = {}
        for name in CUBE_NAMES:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free"
            )
            self._cube_joint_addresses[name] = int(self.model.jnt_qposadr[jid])
            self._cube_dof_addresses[name] = int(self.model.jnt_dofadr[jid])
            self._cube_geom_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom"
            )
            self._cube_body_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )

        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self.goal = np.array([0.30, 0.07], np.float32)
        self._rotation: np.ndarray | None = None
        self._substep_phase = 0
        self.reset(seed)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[self._joint_qpos[name]] for name in JOINTS.names],
            dtype=np.float32,
        )

    @property
    def ee_xy(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id][:2].copy()

    def true_object_xy(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids["red_cube"]][:2].copy()

    def write_cube_pose(self, name: str, position: np.ndarray) -> None:
        addr = self._cube_joint_addresses[name]
        self.data.qpos[addr : addr + 3] = position
        self.data.qpos[addr + 3 : addr + 7] = (1.0, 0.0, 0.0, 0.0)
        dof = self._cube_dof_addresses[name]
        self.data.qvel[dof : dof + 6] = 0.0

    def reset(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        mujoco.mj_resetData(self.model, self.data)

        home = JOINTS.home_array()
        for i, name in enumerate(JOINTS.names):
            self.data.qpos[self._joint_qpos[name]] = home[i]
            self.data.ctrl[self._actuator_ids[name]] = home[i]

        box_xy = np.asarray((rng.uniform(0.28, 0.34), rng.uniform(0.10, 0.17)))
        points: list[np.ndarray] = []
        for name in CUBE_NAMES:
            for _ in range(500):
                p = np.asarray(
                    (rng.uniform(0.17, 0.32), rng.uniform(-0.16, 0.06))
                )
                if (
                    np.linalg.norm(p - box_xy) >= 0.11
                    and all(np.linalg.norm(p - q) >= 0.052 for q in points)
                ):
                    points.append(p)
                    break
            else:
                raise RuntimeError("scene randomization failed")
            self.write_cube_pose(
                name, np.array([points[-1][0], points[-1][1], self.cube_half_size])
            )

        self.data.mocap_pos[0] = (box_xy[0], box_xy[1], 0.006)
        self.data.mocap_quat[0] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        self._substep_phase = 0

    def hide_distractors(self) -> None:
        for name, xy in {
            "green_cube": (0.45, 0.30),
            "yellow_cube": (0.46, -0.30),
            "purple_cube": (0.48, 0.00),
        }.items():
            self.write_cube_pose(
                name, np.array([xy[0], xy[1], self.cube_half_size])
            )
        self.data.mocap_pos[0] = (0.46, 0.28, 0.006)
        mujoco.mj_forward(self.model, self.data)

    def set_joints_direct(self, joints: np.ndarray) -> None:
        for i, name in enumerate(JOINTS.names):
            self.data.qpos[self._joint_qpos[name]] = joints[i]
            self.data.ctrl[self._actuator_ids[name]] = joints[i]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

    def solve_ik(
        self,
        target: np.ndarray,
        target_rotation: np.ndarray,
        max_iters: int = 220,
    ) -> tuple[bool, np.ndarray, float]:
        work = self._ik_work
        joints = self.joint_positions.astype(float)
        lower = JOINTS.lower_array()[:5].astype(float) + 0.01
        upper = JOINTS.upper_array()[:5].astype(float) - 0.01
        jac_pos = np.zeros((3, self.model.nv), dtype=float)
        jac_rot = np.zeros((3, self.model.nv), dtype=float)
        orientation_weight = 0.12

        for _ in range(max_iters):
            for i, name in enumerate(JOINTS.names[:5]):
                work.qpos[self._joint_qpos[name]] = joints[i]
            work.qvel[:] = 0.0
            mujoco.mj_forward(self.model, work)

            error = np.asarray(target, float) - work.site_xpos[self._site_id]
            norm = float(np.linalg.norm(error))
            current_rotation = work.site_xmat[self._site_id].reshape(3, 3)
            rotation_delta = target_rotation @ current_rotation.T
            rotation_error = 0.5 * np.asarray(
                (
                    rotation_delta[2, 1] - rotation_delta[1, 2],
                    rotation_delta[0, 2] - rotation_delta[2, 0],
                    rotation_delta[1, 0] - rotation_delta[0, 1],
                )
            )
            if norm <= 0.003:
                out = self.joint_positions.astype(float)
                out[:5] = joints[:5]
                return True, out, norm

            mujoco.mj_jacSite(
                self.model, work, jac_pos, jac_rot, self._site_id
            )
            cols = [
                int(self.model.jnt_dofadr[self._joint_ids[name]])
                for name in JOINTS.names[:5]
            ]
            J = np.vstack(
                (jac_pos[:, cols], orientation_weight * jac_rot[:, cols])
            )
            combined = np.concatenate(
                (error, orientation_weight * rotation_error)
            )
            damped = J @ J.T + 0.0025 * np.eye(6)
            delta = J.T @ np.linalg.solve(damped, combined)
            joints[:5] = np.clip(
                joints[:5] + np.clip(delta, -0.08, 0.08), lower, upper
            )

        out = self.joint_positions.astype(float)
        out[:5] = joints[:5]
        for i, name in enumerate(JOINTS.names[:5]):
            work.qpos[self._joint_qpos[name]] = joints[i]
        mujoco.mj_forward(self.model, work)
        final = float(
            np.linalg.norm(np.asarray(target) - work.site_xpos[self._site_id])
        )
        return final <= 0.01, out, final

    def _physics_substeps(self) -> int:
        pattern = (16, 17, 17)
        value = pattern[self._substep_phase % len(pattern)]
        self._substep_phase += 1
        return value

    def _joint_trajectory(self, target: np.ndarray, steps: int = 2) -> None:
        start = self.joint_positions.astype(float)
        for fraction in np.linspace(1.0 / steps, 1.0, steps):
            q = start + fraction * (target - start)
            for i, name in enumerate(JOINTS.names):
                self.data.ctrl[self._actuator_ids[name]] = q[i]
            for _ in range(self._physics_substeps()):
                mujoco.mj_step(self.model, self.data)

    def reset_arm_to_xy(
        self, xy: np.ndarray, preserve_object: bool = True
    ) -> bool:
        opened = self.joint_positions.astype(float)
        opened[-1] = 1.55
        self.set_joints_direct(opened)
        self._rotation = self.data.site_xmat[self._site_id].reshape(3, 3).copy()

        ok, joints, _ = self.solve_ik(
            np.array([xy[0], xy[1], self.push_z]), self._rotation
        )
        if not ok:
            return False
        joints[-1] = 1.55

        saved = None
        addr = self._cube_joint_addresses["red_cube"]
        if preserve_object:
            saved = self.data.qpos[addr : addr + 7].copy()

        self.set_joints_direct(joints)

        if saved is not None:
            self.data.qpos[addr : addr + 7] = saved
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        return True

    def step(self, action_xy: np.ndarray) -> np.ndarray:
        if self._rotation is None:
            self._rotation = self.data.site_xmat[self._site_id].reshape(3, 3).copy()

        action = np.clip(
            np.asarray(action_xy, np.float32),
            -self.action_limit,
            self.action_limit,
        )
        target_xy = self.ee_xy + action
        target_xy[0] = np.clip(target_xy[0], *self.xlim)
        target_xy[1] = np.clip(target_xy[1], *self.ylim)

        ok, joints, _ = self.solve_ik(
            np.array([target_xy[0], target_xy[1], self.push_z]),
            self._rotation,
        )
        if not ok:
            joints = self.joint_positions.astype(float)
        joints[-1] = 1.55
        self._joint_trajectory(joints)
        return self.state_without_object()

    def state_without_object(self) -> np.ndarray:
        return np.concatenate(
            [
                self.ee_xy.astype(np.float32),
                self.joint_positions[:5].astype(np.float32),
            ]
        )
