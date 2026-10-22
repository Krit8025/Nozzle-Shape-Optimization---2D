"""
Optimization Runner Script using Differential Evolution.
Optimizes a supersonic nozzle contour for maximum thrust coefficient C_T
at a specified length L, compares with baseline, and generates diagnostic plots.
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thermo import GasDynamics
from geometry import BezierWall, ParabolicWall
from moc_solver import GeneralMoCSolver
from optimizer import NozzleOptimizer
from visualization import MoCVisualizer


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    gas = GasDynamics(gamma=1.4, R=287.05)

    # -------------------------------------------------------------------------
    # Optimization Problem Definition
    # -------------------------------------------------------------------------
    nozzle_length = 5.0       # Fixed length L from throat to exit (m)
    y_throat = 1.0            # Throat half-height yt (m)
    p0 = 10.0e5               # 10 bar chamber pressure (Pa)
    T0 = 3000.0               # 3000 K chamber temperature (K)
    p_amb = 101325.0          # 1 atm ambient pressure (Pa) (sea-level operation)

    print("\n" + "=" * 78)
    print("      STEP 3: NOZZLE CONTOUR OPTIMIZATION VIA DIFFERENTIAL EVOLUTION")
    print("=" * 78)
    print(f"Design Constraint       : Fixed Nozzle Length L = {nozzle_length:.2f} m")
    print(f"Throat Half-Height (yt) : {y_throat:.2f} m")
    print(f"Operating Conditions    : p0 = {p0/1e5:.1f} bar, T0 = {T0:.0f} K, pamb = {p_amb/1e5:.3f} bar")
    print(f"Optimizer Selected      : Differential Evolution (scipy.optimize.differential_evolution)")
    print("-" * 78)

    # -------------------------------------------------------------------------
    # 1. Baseline Benchmark Nozzle (Standard Divergent Bell)
    # -------------------------------------------------------------------------
    print("\n[1/4] Evaluating Baseline Unoptimized Nozzle Contour...")
    P0_base = (0.0, y_throat)
    P1_base = (1.5, 1.6)
    P2_base = (3.2, 2.1)
    P3_base = (nozzle_length, 2.3)
    baseline_wall = BezierWall(P0_base, P1_base, P2_base, P3_base)

    solver = GeneralMoCSolver(gas)
    base_sol = solver.solve(baseline_wall, num_rays=30, p0=p0, T0=T0, p_amb=p_amb)
    base_CT = base_sol.thrust["thrust_coefficient_CT"]
    base_thrust = base_sol.thrust["total_thrust_N"]

    print(f"Baseline Area Ratio     : {baseline_wall.y_exit / y_throat:.4f}")
    print(f"Baseline Planar Thrust  : {base_thrust:.2f} N/m")
    print(f"Baseline Thrust Coeff CT: {base_CT:.5f}")

    # -------------------------------------------------------------------------
    # 2. Run Differential Evolution Optimization
    # -------------------------------------------------------------------------
    print("\n[2/4] Executing Differential Evolution...")
    opt = NozzleOptimizer(
        length=nozzle_length,
        y_throat=y_throat,
        p0=p0,
        T0=T0,
        p_amb=p_amb,
        gas=gas,
        num_rays=18,  # Fast and accurate for population evaluations
    )

    # Run DE with popsize=10, maxiter=20 for rapid global convergence
    opt_results = opt.optimize_differential_evolution(
        y_exit_bounds=(1.8, 3.0),
        th0_bounds=(18.0, 34.0),
        the_bounds=(1.0, 10.0),
        maxiter=15,
        popsize=8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        verbose=True,
    )

    opt_wall = opt_results["opt_wall"]
    opt_sol = opt_results["opt_solution"]
    opt_CT = opt_results["optimal_CT"]
    opt_thrust = opt_results["optimal_thrust_N"]

    # -------------------------------------------------------------------------
    # 3. Performance Comparison Table
    # -------------------------------------------------------------------------
    print("\n[3/4] Performance Comparison: Baseline vs. Differential Evolution Optimal")
    print("=" * 78)
    print(f"{'Performance Metric':<30} {'Baseline':<20} {'DE Optimal':<20} {'Improvement':<10}")
    print("-" * 78)

    delta_CT_pct = (opt_CT - base_CT) / base_CT * 100.0
    delta_F_pct = (opt_thrust - base_thrust) / base_thrust * 100.0

    print(f"{'Thrust Coefficient (CT)':<30} {base_CT:<20.5f} {opt_CT:<20.5f} {delta_CT_pct:+6.2f} %")
    print(f"{'Total Planar Thrust (N/m)':<30} {base_thrust:<20.2f} {opt_thrust:<20.2f} {delta_F_pct:+6.2f} %")
    print(f"{'Exit Half-Height (m)':<30} {baseline_wall.y_exit:<20.4f} {opt_wall.y_exit:<20.4f} -")
    print(f"{'Exit Area Ratio (Ae/At)':<30} {baseline_wall.y_exit/y_throat:<20.4f} {opt_wall.y_exit/y_throat:<20.4f} -")
    print(f"{'Mean Exit Mach':<30} {base_sol.thrust['average_exit_Mach']:<20.3f} {opt_sol.thrust['average_exit_Mach']:<20.3f} -")
    print(f"{'Mass Flow Rate (kg/s)':<30} {base_sol.thrust['mass_flow_rate_kg_s']:<20.2f} {opt_sol.thrust['mass_flow_rate_kg_s']:<20.2f} -")
    print(f"{'Divergence Efficiency (%)':<30} {base_sol.thrust['divergence_efficiency']*100.0:<20.2f} {opt_sol.thrust['divergence_efficiency']*100.0:<20.2f} -")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 4. Generate Publication-Quality Plots
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Diagnostic & Comparison Figures...")

    # Plot A: DE Convergence History
    if opt_results["history_gen"]:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        gens = [g["generation"] for g in opt_results["history_gen"]]
        cts = [g["best_CT"] for g in opt_results["history_gen"]]
        ax.plot(gens, cts, "o-", color="navy", lw=2.2, markersize=6)
        ax.axhline(base_CT, color="crimson", linestyle="--", lw=1.5, label=f"Baseline $C_T = {base_CT:.4f}$")
        ax.set_xlabel("Differential Evolution Generation", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best Thrust Coefficient, $C_T$", fontsize=11, fontweight="bold")
        ax.set_title("Differential Evolution Optimization Convergence History", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        conv_path = os.path.join(output_dir, "optimization_convergence.png")
        plt.savefig(conv_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Convergence history plot saved: {conv_path}")

    # Plot B: Baseline vs. Optimized Contour Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    x_dense = np.linspace(0.0, nozzle_length, 250)

    y_base = [baseline_wall.y(x) for x in x_dense]
    y_opt = [opt_wall.y(x) for x in x_dense]

    # Upper walls
    ax.plot(x_dense, y_base, "--", color="crimson", lw=2.2, label=f"Baseline Contour ($C_T = {base_CT:.4f}$)")
    ax.plot(x_dense, y_opt, "-", color="navy", lw=2.8, label=f"DE Optimal Contour ($C_T = {opt_CT:.4f}$)")

    # Lower symmetric walls
    ax.plot(x_dense, [-y for y in y_base], "--", color="crimson", lw=2.2)
    ax.plot(x_dense, [-y for y in y_opt], "-", color="navy", lw=2.8)

    # Control points polygon for optimal contour
    ctrl_x = [opt_wall.P0[0], opt_wall.P1[0], opt_wall.P2[0], opt_wall.P3[0]]
    ctrl_y = [opt_wall.P0[1], opt_wall.P1[1], opt_wall.P2[1], opt_wall.P3[1]]
    ax.plot(ctrl_x, ctrl_y, "go--", alpha=0.5, label="Optimal Bezier Control Polygon")

    ax.axhline(0, color="gray", linestyle="-.", lw=1.2, label="Centerline")
    ax.set_title(
        f"Supersonic Nozzle Shape Optimization (Fixed Length $L = {nozzle_length:.1f}$ m)\n"
        f"Thrust Gain: +{delta_F_pct:.2f}% | Optimal Area Ratio: {opt_wall.y_exit/y_throat:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Axial Position, $x$ (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Vertical Position, $y$ (m)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper left")

    plt.tight_layout()
    contour_path = os.path.join(output_dir, "optimization_contour_comparison.png")
    plt.savefig(contour_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Contour comparison plot saved   : {contour_path}")

    # Plot C: Full Flowfield on Optimal Nozzle
    vis = MoCVisualizer()
    flow_path = os.path.join(output_dir, "optimization_optimal_flowfield.png")
    vis.plot_flowfield(opt_sol, property_name="M", save_path=flow_path)
    print(f"[OK] Optimal flowfield plot saved    : {flow_path}")

    print("\n" + "=" * 78)
    print("                STEP 3 OPTIMIZATION COMPLETE!                      ")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

