"""Numerical optimization of IK config parameters.

Provides derivative-free optimization of ``human_scale_table`` (and optionally
``human_height_assumption``) by repeatedly retargeting a T-pose frame and
minimising the total position deviation against a T-pose spec.

Callers pre-load the GMR retargeter and MuJoCo model once, then pass
lightweight callbacks into ``optimize_scales`` so each evaluation avoids
expensive re-initialisation.

Requires ``scipy`` (already a dependency via ``compute_direct_patch``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RetargetFn = Callable[[dict, float], np.ndarray]
"""``retarget_fn(scale_table, height) -> qpos`` — re-retarget with new params."""

DeviationFn = Callable[[np.ndarray], float]
"""``deviation_fn(qpos) -> float`` — return total position error in metres."""


def _build_objective(
    bones: list[str],
    init_height: float,
    retarget_fn: RetargetFn,
    deviation_fn: DeviationFn,
    optimize_height: bool,
    on_eval: Callable[[int, float], None] | None = None,
) -> Callable[[np.ndarray], float]:
    """Build the scalar objective for scipy.optimize."""
    n_evals = 0

    def objective(x: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        scale_table = {b: float(v) for b, v in zip(bones, x[: len(bones)], strict=True)}
        height = float(x[len(bones)]) if optimize_height else init_height
        try:
            qpos = retarget_fn(scale_table, height)
        except Exception:
            return 1e6
        err = deviation_fn(qpos)
        if on_eval is not None:
            on_eval(n_evals, err)
        return err

    return objective


def optimize_scales(
    bones: list[str],
    init_scales: list[float] | np.ndarray,
    init_height: float,
    retarget_fn: RetargetFn,
    deviation_fn: DeviationFn,
    optimize_height: bool = False,
    max_evals: int = 80,
    scale_bounds: tuple[float, float] = (0.2, 2.5),
    height_bounds: tuple[float, float] = (0.5, 3.0),
    method: str = "Nelder-Mead",
    verbose: bool = True,
) -> dict:
    """Numerically optimise human_scale_table to minimise T-pose position error.

    Parameters
    ----------
    bones:
        Ordered list of bone names (keys in human_scale_table).
    init_scales:
        Initial scale values, same order as *bones*.
    init_height:
        Initial ``human_height_assumption`` value.
    retarget_fn:
        ``(scale_table: dict, height: float) -> qpos`` — lightweight callback
        that updates the caller's pre-loaded retargeter attributes and calls
        ``.retarget()`` on a cached T-pose frame. Must be cheap (~0.1s).
    deviation_fn:
        ``(qpos: ndarray) -> float`` — lightweight callback that re-uses a
        pre-loaded MuJoCo model/data, sets qpos, calls mj_forward, and
        returns ``total_position_deviation``. Must be cheap (~0.01s).
    optimize_height:
        If True, also optimise ``human_height_assumption`` alongside scales.
    max_evals:
        Maximum number of objective function evaluations (default 80).
    scale_bounds:
        ``(lo, hi)`` clamp for each scale value.
    height_bounds:
        ``(lo, hi)`` clamp for height (used only if *optimize_height* is True).
    method:
        scipy optimiser name. ``"Nelder-Mead"`` (default) is derivative-free
        and robust for noisy objectives.
    verbose:
        If True, print progress after each evaluation.

    Returns
    -------
    A patch dict ready for ``apply_patch()``:
    ``{"human_scale_table": {bone: value, ...}, ...}``.
    """
    from scipy.optimize import minimize

    init_scales_arr = np.asarray(init_scales, dtype=np.float64)
    n_bones = len(bones)

    x0_parts: list[np.ndarray] = [init_scales_arr]
    if optimize_height:
        x0_parts.append(np.asarray([init_height], dtype=np.float64))
    x0 = np.concatenate(x0_parts)

    def _on_eval(n: int, err: float) -> None:
        if verbose and n % 10 == 0:
            print(f"  [opt] eval {n:4d}  pos_err={err * 100:.2f} cm")

    objective = _build_objective(
        bones, init_height, retarget_fn, deviation_fn, optimize_height, _on_eval
    )

    opts: dict = {"maxfev": max_evals, "xatol": 1e-4, "fatol": 1e-5, "adaptive": True}

    result = minimize(objective, x0, method=method, options=opts)

    optimal = result.x
    patch: dict = {
        "human_scale_table": {
            b: float(np.clip(optimal[i], scale_bounds[0], scale_bounds[1]))
            for i, b in enumerate(bones)
        }
    }
    if optimize_height:
        h = float(np.clip(optimal[n_bones], height_bounds[0], height_bounds[1]))
        patch["human_height_assumption"] = h

    if verbose:
        print(f"  [opt] done: {result.nfev} evals, success={result.success}")

    return patch
