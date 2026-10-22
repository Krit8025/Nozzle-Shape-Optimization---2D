"""
Nozzle Contour Shape Optimization Module using Differential Evolution.
Optimizes the supersonic diffusing nozzle contour for maximum thrust coefficient C_T
at a specified nozzle length L and operating conditions (p0, T0, p_amb).

Reference: John D. Anderson, Modern Compressible Flow (Chapter 11) & Fernandes et al. (2023).
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import differential_evolution

from thermo import GasDynamics
from geometry import BezierWall, NozzleWall
from moc_solver import GeneralMoCSolver, MoCSolution
from thrust_calculator import ThrustCalculator


class NozzleOptimizer:
    """
    Optimizes 2D planar supersonic nozzle contours using Differential Evolution.
    Guarantees physical convexity (d^2y/dx^2 <= 0) to prevent internal shock generation.
    """

    def __init__(
        self,
        length: float = 5.0,
        y_throat: float = 1.0,
        p0: float = 10.0e5,       # 10 bar chamber pressure
        T0: float = 3000.0,       # 3000 K chamber temperature
        p_amb: float = 101325.0,  # 1 atm ambient pressure (or 0.0 for vacuum)
        gas: Optional[GasDynamics] = None,
        num_rays: int = 20,       # Accurate & fast resolution for DE evaluations
    ):
        self.length = float(length)
        self.y_throat = float(y_throat)
        self.p0 = float(p0)
        self.T0 = float(T0)
        self.p_amb = float(p_amb)
        self.gas = gas if gas is not None else GasDynamics(gamma=1.4)
        self.num_rays = int(num_rays)
        self.solver = GeneralMoCSolver(self.gas)

        # Optimization logging
        self.eval_count = 0
        self.best_CT = -1.0e9
        self.best_wall: Optional[BezierWall] = None
        self.history_CT: List[float] = []
        self.history_gen: List[Dict] = []
        self._current_gen = 0

    def decode_convex_bell(
        self,
        z: np.ndarray,
        y_exit_bounds: Tuple[float, float],
    ) -> BezierWall:
        """
        Maps optimization parameters z to a strictly convex (d^2y/dx^2 <= 0)
        Thrust Optimized Contour (TOC) supersonic bell nozzle.
        
        Parameters:
            z[0] (th0_deg): Initial expansion turn angle at throat (deg), e.g. [15, 35]
            z[1] (the_deg): Exit divergence angle (deg), e.g. [1, 12]
            z[2] (f1): Tangent distance factor for Control Point 1 in (0, 1)
            z[3] (f2): Tangent distance factor for Control Point 2 in (0, 1)
            z[4] (y_e): Exit half-height within y_exit_bounds
        """
        th0_deg, the_deg, f1, f2, y_e = z

        # Enforce physical rule: exit angle must be smaller than throat angle (divergent turning)
        the_deg = min(the_deg, th0_deg - 2.0)
        th0 = math.radians(th0_deg)
        the = math.radians(the_deg)

        # Chord from throat P0 to exit P3
        P0 = np.array([0.0, self.y_throat])
        P3 = np.array([self.length, float(y_e)])

        # Intermediate control points along initial and final tangent slopes
        # Distance scaling based on nozzle length
        d1 = (0.15 + 0.40 * f1) * self.length
        d2 = (0.15 + 0.40 * f2) * self.length

        P1 = P0 + np.array([d1 * math.cos(th0), d1 * math.sin(th0)])
        P2 = P3 - np.array([d2 * math.cos(the), d2 * math.sin(the)])

        # Enforce strictly monotonic x: 0 < x1 < x2 < L
        if P1[0] <= P0[0] + 0.05:
            P1[0] = P0[0] + 0.1
        if P2[0] >= P3[0] - 0.05:
            P2[0] = P3[0] - 0.1
        if P2[0] <= P1[0] + 0.1:
            mid_x = 0.5 * (P1[0] + P2[0])
            P1[0] = mid_x - 0.1
            P2[0] = mid_x + 0.1

        # Enforce vertical monotonic ordering: yt < y1 <= y2 < ye
        P1[1] = max(self.y_throat + 0.01, P1[1])
        P2[1] = min(float(y_e) - 0.01, max(P1[1], P2[1]))

        return BezierWall(
            (float(P0[0]), float(P0[1])),
            (float(P1[0]), float(P1[1])),
            (float(P2[0]), float(P2[1])),
            (float(P3[0]), float(P3[1])),
        )

    def objective(self, z: np.ndarray, y_exit_bounds: Tuple[float, float]) -> float:
        """
        Objective function for Differential Evolution:
        Minimizes -C_T (maximizes planar thrust coefficient).
        """
        self.eval_count += 1
        try:
            wall = self.decode_convex_bell(z, y_exit_bounds)

            # Check geometric convexity: slope must decrease along curve
            # Sample 10 stations along curve
            t_samples = np.linspace(0.0, 1.0, 10)
            slopes = [wall.dydx(wall.get_point(t)[0]) for t in t_samples]
            if any(s < 0.0 for s in slopes):
                return 100.0  # Diverging slope negative -> unphysical

            # Solve flowfield using 2D MoC
            sol = self.solver.solve(
                wall,
                num_rays=self.num_rays,
                p0=self.p0,
                T0=self.T0,
                p_amb=self.p_amb,
            )

            CT = sol.thrust["thrust_coefficient_CT"]

            # Penalize unphysical solutions or numerical errors
            if math.isnan(CT) or math.isinf(CT) or CT < 0.0:
                return 100.0

            if CT > self.best_CT:
                self.best_CT = CT
                self.best_wall = wall

            self.history_CT.append(CT)
            return -CT

        except Exception:
            return 100.0

    def optimize_differential_evolution(
        self,
        y_exit_bounds: Tuple[float, float] = (1.8, 3.0),
        th0_bounds: Tuple[float, float] = (18.0, 34.0),
        the_bounds: Tuple[float, float] = (1.0, 10.0),
        maxiter: int = 20,
        popsize: int = 10,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict:
        """
        Runs Differential Evolution to find the maximum thrust nozzle contour.
        """
        self.eval_count = 0
        self.best_CT = -1.0e9
        self.best_wall = None
        self.history_CT = []
        self.history_gen = []
        self._current_gen = 0

        # Design variables: [th0_deg, the_deg, f1, f2, y_e]
        bounds = [
            (th0_bounds[0], th0_bounds[1]),      # Throat initial angle (deg)
            (the_bounds[0], the_bounds[1]),      # Exit divergence angle (deg)
            (0.0, 1.0),                          # Tangent fraction 1
            (0.0, 1.0),                          # Tangent fraction 2
            (y_exit_bounds[0], y_exit_bounds[1]),# Exit half-height
        ]

        if verbose:
            print("=" * 72)
            print("  STARTING CONVEX BELL DIFFERENTIAL EVOLUTION OPTIMIZATION")
            print("=" * 72)
            print(f"Fixed Length (L)       : {self.length:.3f} m")
            print(f"Throat Half-Height (yt): {self.y_throat:.3f} m")
            print(f"Chamber Pressure (p0)  : {self.p0 / 1e5:.2f} bar")
            print(f"Ambient Pressure (pamb): {self.p_amb / 1e5:.3f} bar")
            print(f"Exit Bounds (ye)       : [{y_exit_bounds[0]:.2f}, {y_exit_bounds[1]:.2f}] m")
            print(f"Angle Bounds           : theta_throat=[{th0_bounds[0]:.1f} deg, {th0_bounds[1]:.1f} deg], theta_exit=[{the_bounds[0]:.1f} deg, {the_bounds[1]:.1f} deg]")
            print(f"DE Settings            : maxiter={maxiter}, popsize={popsize}, seed={seed}")
            print("-" * 72)
            print(f"{'Gen':<6} {'Evaluations':<14} {'Best CT':<14} {'Best Exit ye':<16} {'Time (s)':<10}")
            print("-" * 72)

        start_time = time.time()

        def callback(xk, convergence):
            self._current_gen += 1
            elapsed = time.time() - start_time
            best_ye = float(xk[4])
            self.history_gen.append({
                "generation": self._current_gen,
                "best_CT": self.best_CT,
                "best_ye": best_ye,
                "evaluations": self.eval_count,
                "time_s": elapsed,
            })
            if verbose:
                print(
                    f"{self._current_gen:<6d} {self.eval_count:<14d} {self.best_CT:<14.5f} "
                    f"{best_ye:<16.4f} {elapsed:<10.1f}"
                )

        # Run Differential Evolution
        res = differential_evolution(
            func=self.objective,
            bounds=bounds,
            args=(y_exit_bounds,),
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            callback=callback,
            polish=False,
        )

        total_time = time.time() - start_time
        opt_z = res.x
        opt_wall = self.decode_convex_bell(opt_z, y_exit_bounds)

        # High-resolution final evaluation (N = 35 rays for publication accuracy)
        final_sol = self.solver.solve(
            opt_wall,
            num_rays=35,
            p0=self.p0,
            T0=self.T0,
            p_amb=self.p_amb,
        )

        if verbose:
            print("-" * 72)
            print(f"Optimization Completed in {total_time:.2f} seconds ({self.eval_count} evaluations).")
            print(f"Optimal Thrust Coefficient (CT) : {final_sol.thrust['thrust_coefficient_CT']:.5f}")
            print(f"Optimal Total Thrust (F)        : {final_sol.thrust['total_thrust_N']:.2f} N/m")
            print(f"Optimal Exit Half-Height (ye)   : {opt_wall.y_exit:.4f} m (Area Ratio: {opt_wall.y_exit/self.y_throat:.4f})")
            print(f"Optimal Throat Angle (th0)      : {math.degrees(opt_wall.theta(0.0)):.2f} deg")
            print(f"Optimal Exit Angle (the)        : {math.degrees(opt_wall.theta(self.length)):.2f} deg")
            print(f"Optimal Control Points: P0={opt_wall.P0.tolist()}, P1={np.round(opt_wall.P1, 4).tolist()}, "
                  f"P2={np.round(opt_wall.P2, 4).tolist()}, P3={np.round(opt_wall.P3, 4).tolist()}")
            print("=" * 72)

        return {
            "success": res.success,
            "message": res.message,
            "total_evaluations": self.eval_count,
            "total_time_s": total_time,
            "opt_z": opt_z,
            "opt_wall": opt_wall,
            "opt_solution": final_sol,
            "history_gen": self.history_gen,
            "optimal_CT": final_sol.thrust["thrust_coefficient_CT"],
            "optimal_thrust_N": final_sol.thrust["total_thrust_N"],
            "optimal_exit_Mach": final_sol.thrust["average_exit_Mach"],
            "optimal_area_ratio": opt_wall.y_exit / self.y_throat,
        }
