from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from .env import SO101PushEnv
from .planner import residual_cem, setup_action
from .vision import (
    GlobalBootstrapDetector,
    WristAppearanceTracker,
    configure_offset_wrist_camera,
)
from .world_model import train_world_model


OUTPUT_DIR = Path("outputs")
EP_LEN = 28


@dataclass
class RunResult:
    start_cm: float
    end_cm: float
    best_cm: float
    steps: int
    success: bool
    gif_path: Path


class TinyWorldBot:
    """SO-101 pushing with one-shot global vision + wrist contact vision."""

    palette = [
        (0.24, 0.42, 0.68, 1.0),
        (0.62, 0.43, 0.18, 1.0),
        (0.32, 0.62, 0.35, 1.0),
        (0.58, 0.28, 0.62, 1.0),
        (0.12, 0.58, 0.62, 1.0),
    ]

    def __init__(self, seed: int = 0) -> None:
        self.env = SO101PushEnv(seed=seed)
        configure_offset_wrist_camera(self.env)
        self.global_detector = GlobalBootstrapDetector(self.env)
        self.wrist = WristAppearanceTracker(self.env)
        self.rng = np.random.default_rng(seed)
        self.global_xy: np.ndarray | None = None
        self.last_obj: np.ndarray | None = None
        self.wrist_active = False
        self.wrist_switches = 0

    def _set_appearance(self, training: bool) -> None:
        gid = self.env._cube_geom_ids["red_cube"]
        if training:
            color = self.palette[int(self.rng.integers(0, len(self.palette)))]
        else:
            # Held-out mustard color; never appears in training.
            color = (0.55, 0.50, 0.18, 1.0)
        self.env.model.geom_rgba[gid] = color

    def _reset_perception(self) -> None:
        self.wrist.reset()
        self.wrist_active = False
        self.last_obj = None

    def object_xy(self) -> np.ndarray:
        if self.global_xy is None:
            raise RuntimeError("episode has not been initialized")

        if not self.wrist_active:
            obj = self.global_xy.copy()
            self.last_obj = obj
            if np.linalg.norm(self.env.ee_xy - obj) < 0.067:
                if self.wrist.bootstrap(obj):
                    self.wrist_active = True
                    self.wrist_switches += 1
            return obj

        obj = self.wrist.track()
        self.last_obj = obj.copy()
        return obj

    def state(self) -> np.ndarray:
        obj = self.object_xy()
        return np.concatenate(
            [
                self.env.ee_xy.astype(np.float32),
                obj.astype(np.float32),
                self.env.joint_positions[:5].astype(np.float32),
            ]
        )

    def reset_episode(self, training: bool) -> np.ndarray:
        for _ in range(12):
            self.env.reset(int(self.rng.integers(0, 2**31 - 1)))
            self.env.hide_distractors()
            self._set_appearance(training)

            if not training:
                self.env.write_cube_pose(
                    "red_cube",
                    np.array([0.22, -0.03, self.env.cube_half_size]),
                )
                self.env.goal = np.array([0.30, 0.07], np.float32)
                mujoco.mj_forward(self.env.model, self.env.data)

            # The global camera is used exactly once, before the robot moves.
            try:
                self.global_xy = self.global_detector.locate()
            except RuntimeError:
                continue

            self._reset_perception()
            self.last_obj = self.global_xy.copy()

            if training:
                theta = self.rng.uniform(-math.pi, math.pi)
                radius = self.rng.uniform(0.052, 0.078)
                start = self.global_xy + radius * np.array(
                    [math.cos(theta), math.sin(theta)]
                )
            else:
                # Deliberately start on the wrong side.
                start = self.global_xy + np.array([0.055, 0.045])

            start[0] = np.clip(start[0], *self.env.xlim)
            start[1] = np.clip(start[1], *self.env.ylim)
            if self.env.reset_arm_to_xy(start, preserve_object=True):
                return self.state()

        raise RuntimeError("failed to initialize episode")

    def exploration_action(self, state: np.ndarray) -> np.ndarray:
        ee, obj = state[:2], state[2:4]
        if self.rng.random() < 0.72:
            action = obj - ee
            norm = np.linalg.norm(action)
            if norm > 1e-6:
                action /= norm
            action += self.rng.normal(0, 0.65, size=2)
            action *= self.rng.uniform(0.008, self.env.action_limit)
        else:
            action = self.rng.uniform(
                -self.env.action_limit, self.env.action_limit, size=2
            )
        return np.clip(
            action, -self.env.action_limit, self.env.action_limit
        ).astype(np.float32)

    def collect(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.empty((count, 9), np.float32)
        a = np.empty((count, 2), np.float32)
        ns = np.empty((count, 9), np.float32)

        self.reset_episode(training=True)
        for i in range(count):
            if i % EP_LEN == 0:
                self.reset_episode(training=True)

            state = self.state()
            action = self.exploration_action(state)
            self.env.step(action)
            next_state = self.state()
            s[i], a[i], ns[i] = state, action, next_state

            if (i + 1) % 250 == 0:
                print(
                    f"collected {i+1}/{count} "
                    f"wrist_switches={self.wrist_switches} "
                    f"wrist_failures={self.wrist.failures}"
                )
        return s, a, ns

    def evaluation_distance(self) -> float:
        # Privileged MuJoCo position is isolated to evaluation only.
        return float(np.linalg.norm(self.env.true_object_xy() - self.env.goal))

    @staticmethod
    def _annotate(image, step, mode, tracker, distance, obj):
        im = Image.fromarray(image).resize((640, 480))
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 640, 60), fill=(0, 0, 0))
        draw.text(
            (12, 8),
            f"TinyWorldBot | step {step:02d} | {mode.upper()} | {tracker}",
            fill="white",
        )
        draw.text(
            (12, 32),
            f"eval distance={distance*100:.2f} cm  visual object={obj.round(3)}",
            fill="white",
        )
        return im

    def evaluate(self, model, stats, device, max_steps: int = 85) -> RunResult:
        self.reset_episode(training=False)
        start = self.evaluation_distance()
        best = start
        mode = "setup"
        goal = self.env.goal.copy()

        state = self.state().copy()
        s1 = state.copy()
        s2 = state.copy()
        a1 = np.zeros(2, np.float32)
        a2 = np.zeros(2, np.float32)

        renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        frames: list[Image.Image] = []

        for t in range(max_steps):
            obj = state[2:4]
            if mode == "setup":
                action, ready = setup_action(self.env.ee_xy, obj, goal)
                if ready:
                    if not self.wrist_active and self.wrist.bootstrap(obj):
                        self.wrist_active = True
                        self.wrist_switches += 1
                    mode = "push"
            else:
                action = residual_cem(
                    model, stats, state, s1, s2, a1, a2, goal, device
                )

            self.env.step(action)
            post = self.state()
            post_obj = post[2:4]
            distance = self.evaluation_distance()
            best = min(best, distance)

            if mode == "push" and (t + 1) % 6 == 0:
                direction = goal - post_obj
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                    rel = self.env.ee_xy - post_obj
                    if (
                        np.dot(rel, direction) > 0.004
                        or np.linalg.norm(rel) > 0.072
                    ):
                        mode = "setup"
                        self.global_xy = post_obj.copy()
                        self.wrist_active = False

            tracker = "WRIST" if self.wrist_active else "GLOBAL-ONCE"
            frame = (
                self.wrist.render()
                if self.wrist_active
                else self.global_detector.render()
            )
            frames.append(
                self._annotate(frame, t+1, mode, tracker, distance, post_obj)
            )

            if t == 0 or (t + 1) % 5 == 0:
                print(
                    f"step {t+1:02d} mode={mode:<5} tracker={tracker:<11} "
                    f"object={np.round(post_obj, 3)} "
                    f"distance={distance*100:.1f}cm"
                )

            if distance < 0.04:
                break

            s2, s1, state = s1, state, post
            a2, a1 = a1, np.asarray(action, np.float32)

        renderer.close()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gif_path = OUTPUT_DIR / "demo.gif"
        if frames:
            hold = [frames[0]]*4 + frames[1:-1] + [frames[-1]]*7
            hold[0].save(
                gif_path,
                save_all=True,
                append_images=hold[1:],
                duration=160,
                loop=0,
            )

        end = self.evaluation_distance()
        return RunResult(
            start_cm=start*100,
            end_cm=end*100,
            best_cm=best*100,
            steps=t+1,
            success=end < 0.04,
            gif_path=gif_path,
        )

    def close(self) -> None:
        self.global_detector.close()
        self.wrist.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and run TinyWorldBot in MuJoCo."
    )
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    bot = TinyWorldBot(seed=args.seed)
    try:
        s, a, ns = bot.collect(args.samples)
        print("training history-aware world model...")
        model, stats = train_world_model(
            s, a, ns, device=device, epochs=args.epochs
        )
        print("evaluating held-out object appearance...")
        result = bot.evaluate(model, stats, device)
    finally:
        bot.close()

    print(
        f"RESULT start={result.start_cm:.2f}cm "
        f"end={result.end_cm:.2f}cm "
        f"best={result.best_cm:.2f}cm "
        f"steps={result.steps} success={result.success}"
    )
    print("simulator object XY used by controller/training: NO")
    print("hard-coded object color threshold: NO")
    print("global vision: one-shot current-vs-empty RGB")
    print("contact vision: side-offset wrist camera + learned appearance ROI")
    print("gif:", result.gif_path)


if __name__ == "__main__":
    main()
