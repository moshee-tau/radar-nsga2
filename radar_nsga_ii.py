#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II (pymoo) live visualization + GIF/MP4 export for the Radar multi-objective problem.

Run:
  python nsga2_radar_live.py

Dependencies:
  pip install pymoo numpy matplotlib pillow

Optional (for MP4):
  Install ffmpeg on your system, then set SAVE_MP4=True in main() or via CLI.

Notes:
  - We solve a min-min problem with f1 = -R (so maximizing R), f2 = Cost (minimize).
  - We visualize the "search" live per generation and save an animation at the end.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RadarConsts:
    """@brief Physical/constants used by the radar range equation."""
    eta: float = 0.8
    lam: float = 0.1
    k: float = 1.380662e-23
    T: float = 290.0
    B: float = 10e6
    Fn: float = 3.16
    SNR: float = 20.0


@dataclass
class Bounds:
    """@brief Decision-variable bounds for Pt, A, sigma."""
    Pt_min: float = 1000.0
    Pt_max: float = 50000.0
    A_min: float = 0.5
    A_max: float = 12.0
    sigma_min: float = 1.5
    sigma_max: float = 5.0


@dataclass
class Constraints:
    """
    @brief Optional feasibility constraints.
    @param R_min      Enforce R >= R_min if not None.
    @param cost_limit Enforce Cost <= cost_limit if not None.
    """
    R_min: Optional[float] = None
    cost_limit: Optional[float] = None


@dataclass
class NSGA2Params:
    """@brief Algorithm parameters for NSGA-II."""
    pop_size: int = 200
    n_gen: int = 200
    seed: int = 1
    eliminate_duplicates: bool = True


@dataclass
class VizParams:
    """
    @brief Visualization and export parameters.
    @param every          Record/plot every N generations (1 = every generation).
    @param live           Show a live updating window during optimization.
    @param save_gif       Save GIF animation at the end.
    @param save_mp4       Save MP4 animation at the end (requires ffmpeg).
    @param out_gif        GIF output path.
    @param out_mp4        MP4 output path.
    @param interval_ms    Frame interval for animation playback.
    """
    every: int = 1
    live: bool = True
    save_gif: bool = True
    save_mp4: bool = False
    out_gif: str = "nsga2_search.gif"
    out_mp4: str = "nsga2_search.mp4"
    interval_ms: int = 120


# =============================================================================
# Model: radar range + cost
# =============================================================================

def radar_range(Pt: float, A: float, sigma: float, c: RadarConsts) -> float:
    """
    @brief Compute radar detection range R (meters) from decision variables.
    @details
      R = ( Pt * eta * A^2 * sigma / (4*pi*lam^2*k*T*B*Fn*SNR) )^(1/4)
    @param Pt     Transmit power [W]
    @param A      Antenna aperture area [m^2]
    @param sigma  Radar cross section [m^2]
    @param c      Radar constants
    @return Range R [m]
    """
    denom = (4.0 * np.pi) * (c.lam ** 2) * c.k * c.T * c.B * c.Fn * c.SNR
    # Guard for numerical stability
    val = (Pt * c.eta * (A ** 2) * sigma) / denom
    if val <= 0:
        return 0.0
    return float(val ** 0.25)


def radar_cost(Pt: float, A: float) -> float:
    """
    @brief Simple cost model.
    @details COST = 10*Pt + 100000*A
    @param Pt Transmit power [W]
    @param A  Antenna aperture area [m^2]
    @return Cost (arbitrary currency units)
    """
    return float(10.0 * Pt + 100000.0 * A)


# =============================================================================
# pymoo problem definition
# =============================================================================

class RadarMOO(ElementwiseProblem):
    """
    @brief Multi-objective radar design problem for NSGA-II (pymoo).
    @details
      Decision variables:
        x[0] = Pt    in [Pt_min, Pt_max]
        x[1] = A     in [A_min, A_max]
        x[2] = sigma in [sigma_min, sigma_max]

      Objectives (min-min form):
        f1 = -R      (equivalent to maximize R)
        f2 = COST    (minimize cost)

      Constraints (optional, <= 0):
        g1 = R_min - R          (enforces R >= R_min)
        g2 = COST - cost_limit  (enforces COST <= cost_limit)
    """

    def __init__(self, consts: RadarConsts, bounds: Bounds, cons: Constraints):
        self.consts = consts
        # self.bounds = bounds
        self.cons = cons

        n_constr = int(cons.R_min is not None) + int(cons.cost_limit is not None)

        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=n_constr,
            xl=np.array([bounds.Pt_min, bounds.A_min, bounds.sigma_min], dtype=float),
            xu=np.array([bounds.Pt_max, bounds.A_max, bounds.sigma_max], dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        Pt, A, sigma = float(x[0]), float(x[1]), float(x[2])

        R = radar_range(Pt, A, sigma, self.consts)
        C = radar_cost(Pt, A)

        # min-min objectives: f1=-R, f2=Cost
        out["F"] = np.array([-R, C], dtype=float)

        g = []
        if self.cons.R_min is not None:
            g.append(self.cons.R_min - R)
        if self.cons.cost_limit is not None:
            g.append(C - self.cons.cost_limit)
        if g:
            out["G"] = np.array(g, dtype=float)


# =============================================================================
# Live visualization and history collection
# =============================================================================

class LiveSearchRecorder:
    """
    @brief Records the population each generation and optionally updates a live plot.

    Usage:
      recorder = LiveSearchRecorder(viz_params)
      res = minimize(..., callback=recorder.callback)

    We store:
      - gen numbers
      - objective arrays F per recorded gen
      - feasibility (optional) per recorded gen
    """

    def __init__(self, viz: VizParams):
        self.viz = viz
        self.gens: List[int] = []
        self.F_hist: List[np.ndarray] = []
        self.feasible_hist: List[Optional[np.ndarray]] = []

        # live plot state
        self._fig = None
        self._ax = None
        self._scat = None
        self._text = None
        self._last_draw = 0.0

    def _init_live_plot(self):
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.set_xlabel("Cost")
        self._ax.set_ylabel("Range (R)")
        self._ax.set_title("NSGA-II Search (live)")
        self._scat = self._ax.scatter([], [], s=12)
        self._text = self._ax.text(0.02, 0.98, "", transform=self._ax.transAxes,
                                   va="top", ha="left")
        self._fig.tight_layout()
        plt.show(block=False)

    def _update_live_plot(self, gen: int, F: np.ndarray, feasible: Optional[np.ndarray]):
        # Convert back to meaningful metrics
        R = -F[:, 0]
        C = F[:, 1]
        pts = np.column_stack([C, R])

        self._scat.set_offsets(pts)

        # Autoscale a bit (robustly)
        if len(C) > 0:
            xmin, xmax = float(np.min(C)), float(np.max(C))
            ymin, ymax = float(np.min(R)), float(np.max(R))
            dx = (xmax - xmin) if xmax > xmin else 1.0
            dy = (ymax - ymin) if ymax > ymin else 1.0
            self._ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)
            self._ax.set_ylim(ymin - 0.05 * dy, ymax + 0.05 * dy)

        if feasible is None:
            self._text.set_text(f"gen={gen} | pop={len(F)}")
        else:
            n_feas = int(np.sum(feasible))
            self._text.set_text(f"gen={gen} | pop={len(F)} | feasible={n_feas}")

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def callback(self, algorithm):
        """@brief pymoo callback invoked every generation."""
        gen = int(getattr(algorithm, "n_gen", -1))

        if gen % self.viz.every != 0:
            return

        pop = algorithm.pop
        F = pop.get("F")
        G = pop.get("G")

        feasible = None
        if G is not None:
            # Feasible if all constraints <= 0
            feasible = np.all(G <= 0, axis=1)

        self.gens.append(gen)
        self.F_hist.append(np.array(F, copy=True))
        self.feasible_hist.append(None if feasible is None else np.array(feasible, copy=True))

        if self.viz.live:
            if self._fig is None:
                self._init_live_plot()

            # Throttle drawing slightly if needed (keeps UI responsive)
            now = time.time()
            if now - self._last_draw >= 0.02:
                self._update_live_plot(gen, F, feasible)
                self._last_draw = now


# =============================================================================
# Animation export
# =============================================================================

def build_animation(
    gens: List[int],
    F_hist: List[np.ndarray],
    feasible_hist: List[Optional[np.ndarray]],
    interval_ms: int = 120,
) -> animation.FuncAnimation:
    """
    @brief Build a Matplotlib animation of the search.
    @return FuncAnimation instance (can be saved as GIF/MP4).
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Cost")
    ax.set_ylabel("Range (R)")
    ax.set_title("NSGA-II Search (animation)")

    scat = ax.scatter([], [], s=12)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    fig.tight_layout()

    # Pre-compute global limits
    allF = np.vstack(F_hist) if len(F_hist) else np.zeros((0, 2))
    allR = -allF[:, 0] if len(allF) else np.array([0.0])
    allC = allF[:, 1] if len(allF) else np.array([0.0])

    xmin, xmax = float(np.min(allC)), float(np.max(allC))
    ymin, ymax = float(np.min(allR)), float(np.max(allR))
    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)
    ax.set_ylim(ymin - 0.05 * dy, ymax + 0.05 * dy)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        text.set_text("")
        return scat, text

    def update(i):
        F = F_hist[i]
        R = -F[:, 0]
        C = F[:, 1]
        scat.set_offsets(np.column_stack([C, R]))

        feas = feasible_hist[i]
        if feas is None:
            text.set_text(f"gen={gens[i]} | pop={len(F)}")
        else:
            text.set_text(f"gen={gens[i]} | pop={len(F)} | feasible={int(np.sum(feas))}")

        ax.set_title("NSGA-II Search (animation)")
        return scat, text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(F_hist),
        init_func=init,
        interval=interval_ms,
        blit=False
    )
    return ani


def save_animation_gif(ani: animation.FuncAnimation, out_path: str, fps: int = 10):
    """
    @brief Save animation as GIF using PillowWriter.
    @note Requires: pip install pillow
    """
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    print(f"[OK] Saved GIF: {out_path}")


def save_animation_mp4(ani: animation.FuncAnimation, out_path: str, fps: int = 20):
    """
    @brief Save animation as MP4 using ffmpeg writer.
    @note Requires ffmpeg installed and discoverable in PATH.
    """
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(out_path, writer=writer)
    print(f"[OK] Saved MP4: {out_path}")


# =============================================================================
# Runner
# =============================================================================

def run_nsga2_with_live_viz(
    consts: RadarConsts,
    bounds: Bounds,
    cons: Constraints,
    nsga: NSGA2Params,
    viz: VizParams,
) -> Tuple[np.ndarray, np.ndarray, LiveSearchRecorder]:
    """
    @brief Run NSGA-II and return (X, F, recorder).
    @return
      X: decision variables (Pareto set)
      F: objective values (Pareto front) where F[:,0]=-R, F[:,1]=Cost
      recorder: recorded history used for animation export
    """
    problem = RadarMOO(consts=consts, bounds=bounds, cons=cons)

    algorithm = NSGA2(
        pop_size=nsga.pop_size,
        eliminate_duplicates=nsga.eliminate_duplicates
    )

    recorder = LiveSearchRecorder(viz=viz)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", nsga.n_gen),
        seed=nsga.seed,
        verbose=True,
        callback=recorder.callback
    )

    # Ensure last frame visible if live
    if viz.live:
        plt.ioff()

    return res.X, res.F, recorder


def print_top_solutions(X: np.ndarray, F: np.ndarray, k: int = 10):
    """
    @brief Print top-K cheapest Pareto solutions (after converting -R back to R).
    """
    R = -F[:, 0]
    C = F[:, 1]
    order = np.argsort(C)
    print(f"\nTop {k} cheapest Pareto solutions:")
    for i in order[:k]:
        Pt, A, sigma = X[i]
        print(f"  Pt={Pt:8.1f} W, A={A:5.2f} m^2, sigma={sigma:4.2f} m^2 | "
              f"R={R[i]/1000:7.2f} km | COST={C[i]:,.0f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSGA-II radar optimization with live visualization + GIF/MP4 export.")
    p.add_argument("--pop", type=int, default=200, help="Population size")
    p.add_argument("--gen", type=int, default=200, help="Number of generations")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--every", type=int, default=1, help="Record/plot every N generations")
    p.add_argument("--no-live", action="store_true", help="Disable live plotting window")
    p.add_argument("--gif", default="nsga2_search.gif", help="Output GIF path")
    p.add_argument("--mp4", default="nsga2_search.mp4", help="Output MP4 path")
    p.add_argument("--save-mp4", action="store_true", help="Save MP4 (requires ffmpeg)")
    p.add_argument("--no-gif", action="store_true", help="Do not save GIF")
    p.add_argument("--Rmin", type=float, default=None, help="Constraint: R >= Rmin (meters).")
    p.add_argument("--Cmax", type=float, default=None, help="Constraint: Cost <= Cmax.")
    return p.parse_args()


def main():
    args = parse_args()

    consts = RadarConsts()
    bounds = Bounds()
    cons = Constraints(R_min=args.Rmin, cost_limit=args.Cmax)

    nsga = NSGA2Params(pop_size=args.pop, n_gen=args.gen, seed=args.seed)
    viz = VizParams(
        every=max(1, int(args.every)),
        live=(not args.no_live),
        save_gif=(not args.no_gif),
        save_mp4=bool(args.save_mp4),
        out_gif=args.gif,
        out_mp4=args.mp4,
        interval_ms=120
    )

    X, F, recorder = run_nsga2_with_live_viz(
        consts=consts,
        bounds=bounds,
        cons=cons,
        nsga=nsga,
        viz=viz
    )

    print_top_solutions(X, F, k=10)

    # Build and save animation from recorded history
    if (viz.save_gif or viz.save_mp4) and len(recorder.F_hist) > 0:
        ani = build_animation(
            gens=recorder.gens,
            F_hist=recorder.F_hist,
            feasible_hist=recorder.feasible_hist,
            interval_ms=viz.interval_ms
        )

        if viz.save_gif:
            save_animation_gif(ani, viz.out_gif, fps=10)

        if viz.save_mp4:
            try:
                save_animation_mp4(ani, viz.out_mp4, fps=20)
            except Exception as e:
                print(f"[WARN] MP4 save failed (ffmpeg likely missing): {e}")

    # Keep final plot window open if live run
    if viz.live:
        plt.show()


if __name__ == "__main__":
    main()
