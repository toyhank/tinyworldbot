from __future__ import annotations

import math

import numpy as np
import torch

from .world_model import predict, torch_feature


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi


def setup_action(ee, obj, goal, max_step: float = 0.018):
    """Move around the object to the side opposite the goal."""
    ee = np.asarray(ee, float)
    obj = np.asarray(obj, float)
    goal = np.asarray(goal, float)

    d = goal - obj
    dn = np.linalg.norm(d)
    if dn < 1e-6:
        return np.zeros(2, np.float32), True
    d /= dn

    desired = math.atan2(-d[1], -d[0])
    rel = ee - obj
    radius = np.linalg.norm(rel)
    current = math.atan2(rel[1], rel[0]) if radius > 1e-6 else desired
    diff = wrap_angle(desired-current)

    clear_radius = 0.074
    setup_radius = 0.052

    if abs(diff) > math.radians(32) and radius < clear_radius-0.006:
        target = obj + clear_radius*np.array(
            [math.cos(current), math.sin(current)]
        )
        ready = False
    elif abs(diff) > math.radians(15):
        da = np.clip(diff, -math.radians(24), math.radians(24))
        angle = current + da
        target = obj + clear_radius*np.array([math.cos(angle), math.sin(angle)])
        ready = False
    else:
        target = obj + setup_radius*np.array(
            [math.cos(desired), math.sin(desired)]
        )
        ready = np.linalg.norm(ee-target) < 0.018

    action = target-ee
    n = np.linalg.norm(action)
    if n > max_step:
        action = action/n*max_step
    return action.astype(np.float32), ready


@torch.inference_mode()
def residual_cem(
    model,
    stats,
    s0_np,
    s1_np,
    s2_np,
    a1_np,
    a2_np,
    goal_np,
    device,
    chunks: int = 5,
    repeat: int = 3,
    population: int = 1400,
    elite: int = 140,
    iters: int = 5,
):
    """CEM over goal-aligned forward/lateral residual controls."""
    s0 = torch.tensor(s0_np, device=device)
    s1_base = torch.tensor(s1_np, device=device)
    s2_base = torch.tensor(s2_np, device=device)
    a1_base = torch.tensor(a1_np, device=device)
    a2_base = torch.tensor(a2_np, device=device)
    goal = torch.tensor(goal_np, device=device)

    g0 = goal-s0[2:4]
    g0 = g0/(torch.linalg.norm(g0)+1e-6)
    lateral = torch.stack([-g0[1], g0[0]])

    mean = torch.zeros(chunks, 2, device=device)
    mean[:, 0] = 0.013
    std = torch.zeros(chunks, 2, device=device)
    std[:, 0] = 0.005
    std[:, 1] = 0.004
    horizon = chunks*repeat

    for _ in range(iters):
        u = mean[None] + std[None]*torch.randn(
            population, chunks, 2, device=device
        )
        u[:, :, 0] = u[:, :, 0].clamp(0.007, 0.018)
        u[:, :, 1] = u[:, :, 1].clamp(-0.009, 0.009)

        action_chunks = (
            u[:, :, 0:1]*g0[None, None, :]
            + u[:, :, 1:2]*lateral[None, None, :]
        )
        actions = action_chunks.repeat_interleave(repeat, dim=1)

        s = s0[None].repeat(population, 1)
        s1 = s1_base[None].repeat(population, 1)
        s2 = s2_base[None].repeat(population, 1)
        a1 = a1_base[None].repeat(population, 1)
        a2 = a2_base[None].repeat(population, 1)

        running = torch.zeros(population, device=device)
        contact = torch.zeros(population, device=device)
        support = torch.zeros(population, device=device)

        for t in range(horizon):
            action = actions[:, t]
            ns, gate = predict(model, stats, s, s1, s2, a1, a2, action)
            obj = ns[:, 2:4]
            ee = ns[:, :2]

            gvec = goal[None]-obj
            gdir = gvec/(torch.linalg.norm(gvec, dim=1, keepdim=True)+1e-6)
            desired_ee = obj-0.020*gdir
            running += 0.060*torch.linalg.norm(obj-goal, dim=1)
            running += 0.14*torch.linalg.norm(ee-desired_ee, dim=1)
            contact += gate

            feat = torch_feature(ns, s, s1, action, a1, a2)
            normalized = (feat-stats["xm"])/stats["xs"]
            support += torch.relu(normalized.abs()-3.2).pow(2).mean(1)

            s2, s1, s = s1, s, ns
            a2, a1 = a1, action

        final = torch.linalg.norm(s[:, 2:4]-goal, dim=1)
        cost = (
            8.0*final
            + running
            + 0.16*support/horizon
            - 0.060*torch.clamp(contact, 0, 8)
        )

        ids = torch.topk(cost, elite, largest=False).indices
        elite_u = u[ids]
        mean = elite_u.mean(0)
        std = elite_u.std(0).clamp(
            torch.tensor([0.0015, 0.0015], device=device),
            torch.tensor([0.006, 0.006], device=device),
        )

    best = mean[0]
    action = best[0]*g0 + best[1]*lateral
    return action.clamp(-0.018, 0.018).cpu().numpy()
