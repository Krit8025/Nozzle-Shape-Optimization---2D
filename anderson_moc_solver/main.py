"""
Main Demonstration and Validation Script for the Clean 2D Planar MoC Solver.
Reference: John D. Anderson, Modern Compressible Flow (Chapter 11) & Fernandes et al. (2023).

Executes:
  1. Golden Reference Benchmark: Anderson Chapter 11 Minimum Length Nozzle (MLN).
  2. Direct Analysis & Thrust Calculation for a given arbitrary diffusing nozzle geometry.
  3. Visualizations and tabular verification.
"""

import os
import math
import numpy as np

from thermo import GasDynamics
from geometry import BezierWall, ParabolicWall, SplineWall
from mln_designer import MinimumLengthNozzle
from moc_solver import GeneralMoCSolver
from visualization import MoCVisualizer


def run_anderson_golden_reference_benchmark():
    """
    Validates the solver against John D. Anderson's Modern Compressible Flow
    Chapter 11, Section 11.5 & Example 11.1 (M_design = 2.4, gamma = 1.4).
    """
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: ANDERSON CH. 11 GOLDEN REFERENCE (MINIMUM LENGTH NOZZLE)")
    print("=" * 70)

    gas = GasDynamics(gamma=1.4)
    M_D = 2.4
    A_exact = gas.get_area_ratio(M_D)
    nu_D = gas.get_nu(M_D)
    theta_max = nu_D / 2.0

    print(f"Target Design Exit Mach (MD) : {M_D:.2f}")
    print(f"Exact Exit Prandtl-Meyer (nu): {math.degrees(nu_D):.3f} deg")
    print(f"Max Expansion Angle (theta_w) : {math.degrees(theta_max):.3f} deg")
    print(f"Exact 1D Isentropic Area Ratio: {A_exact:.4f}")
    print("-" * 70)

    # 1. Anderson coarse grid (N = 7 rays, matching textbook example)
    print("\n--- Running Textbook Case (N = 7 Characteristic Rays) ---")
    mln_coarse = MinimumLengthNozzle(gas)
    res_coarse = mln_coarse.design(M_design=M_D, num_rays=7, y_throat=1.0)

    print(f"Length of Nozzle (L/yt)      : {res_coarse['L_nozzle']:.4f}")
    print(f"Exit Half-Height (ye/yt)     : {res_coarse['y_exit_numerical']:.4f}")
    print(f"Exact 1D Half-Height (ye/yt) : {res_coarse['y_exit_exact']:.4f}")
    print(f"Relative Area Error          : {res_coarse['error_percent']:.4f} %")

    # Display textbook table of wall nodes
    print("\nTextbook Wall Contour Points (Wave Cancellation Section):")
    print(f"{'Node ID':<10} {'x/yt':<10} {'y/yt':<10} {'Mach':<10} {'theta (deg)':<12} {'nu (deg)':<10}")
    print("-" * 65)
    for node in mln_coarse.wall_nodes:
        print(
            f"{str(node.node_id):<10} {node.x:<10.4f} {node.y:<10.4f} {node.M:<10.4f} "
            f"{node.theta_deg:<12.2f} {node.nu_deg:<10.2f}"
        )

    # 2. Grid convergence verification
    print("\n--- Grid Convergence Study (Verifying Convergence to Exact 1D Isentropic) ---")
    print(f"{'Rays (N)':<10} {'ye/yt (MoC)':<15} {'Exact 1D':<15} {'Rel. Error (%)':<15}")
    print("-" * 55)
    for n_rays in [7, 15, 30, 50, 100]:
        mln = MinimumLengthNozzle(gas)
        res = mln.design(M_design=M_D, num_rays=n_rays, y_throat=1.0)
        print(
            f"{n_rays:<10d} {res['y_exit_numerical']:<15.4f} {res['y_exit_exact']:<15.4f} "
            f"{res['error_percent']:<15.4f}"
        )

    # Plot MLN
    output_dir = os.path.dirname(os.path.abspath(__file__))
    mln_fine = MinimumLengthNozzle(gas)
    mln_fine.design(M_design=M_D, num_rays=30, y_throat=1.0)
    vis = MoCVisualizer()
    plot_file = os.path.join(output_dir, "benchmark_anderson_mln.png")
    vis.plot_mln_design(mln_fine, save_path=plot_file)
    print(f"\n[OK] MLN characteristics diagram saved to: {plot_file}")


def run_given_diffusing_nozzle_simulation():
    """
    Demonstrates flowfield solving and thrust calculation for an arbitrary
    diffusing nozzle geometry (Aim 1).
    """
    print("\n" + "=" * 70)
    print("  AIM 1: SOLVE FLOWFIELD AND THRUST FOR A GIVEN DIFFUSING NOZZLE")
    print("=" * 70)

    gas = GasDynamics(gamma=1.4)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Define a contoured supersonic bell nozzle using cubic Bezier curve
    # Throat at (0.0, 1.0), Exit at (5.0, 2.45)
    P0 = (0.0, 1.0)
    P1 = (1.5, 1.7)
    P2 = (3.2, 2.2)
    P3 = (5.0, 2.45)
    wall = BezierWall(P0, P1, P2, P3)

    print(f"Nozzle Type: Cubic Bezier Wall")
    print(f"Control Points: P0={P0}, P1={P1}, P2={P2}, P3={P3}")
    print(f"Throat Wall Angle : {math.degrees(wall.theta(0.0)):.2f} deg")
    print(f"Exit Wall Angle   : {math.degrees(wall.theta(5.0)):.2f} deg")

    # Operating conditions
    p0 = 10.0e5       # 10 bar chamber pressure
    T0 = 3000.0       # 3000 K chamber temperature
    p_amb = 101325.0  # 1 atm sea level ambient pressure

    # Execute MoC solver
    print("\nMarching characteristics through diffusing duct...")
    solver = GeneralMoCSolver(gas)
    sol = solver.solve(wall, num_rays=25, p0=p0, T0=T0, p_amb=p_amb)

    print("\n" + sol.summary())

    # Visualizations
    vis = MoCVisualizer()
    mach_file = os.path.join(output_dir, "flowfield_mach.png")
    vis.plot_flowfield(sol, property_name="M", save_path=mach_file)

    press_file = os.path.join(output_dir, "flowfield_pressure.png")
    vis.plot_flowfield(sol, property_name="p", save_path=press_file)

    print(f"[OK] Mach contour saved to: {mach_file}")
    print(f"[OK] Pressure contour saved to: {press_file}")


if __name__ == "__main__":
    run_anderson_golden_reference_benchmark()
    run_given_diffusing_nozzle_simulation()

