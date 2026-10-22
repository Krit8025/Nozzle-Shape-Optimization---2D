"""
Visualization Module for 2D Planar Method of Characteristics (MoC).
Generates publication-quality figures:
  1. Characteristics Network Plot (C+ and C- Mach waves, walls, centerline)
  2. Mirrored Mach Number Field (Tricontourf)
  3. Wall and Centerline Pressure / Mach Distributions
"""

import math
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from unit_processes import MoCNode
from moc_solver import MoCSolution
from mln_designer import MinimumLengthNozzle


class MoCVisualizer:
    """Renders 2D MoC characteristic grids and flowfields."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

    def plot_mln_design(
        self,
        mln: MinimumLengthNozzle,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """Plots characteristic mesh and wall contour for Anderson MLN design."""
        if not mln.is_designed:
            raise RuntimeError("MLN must be designed before plotting.")

        fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)

        # 1. Plot C- characteristics (expansion rays from throat)
        for i, line in enumerate(mln.c_minus_lines):
            xs = [n.x for n in line]
            ys = [n.y for n in line]
            ax.plot(xs, ys, color="crimson", lw=0.6, alpha=0.6, label="C- waves" if i == 0 else "")
            # Mirror
            ax.plot(xs, [-y for y in ys], color="crimson", lw=0.6, alpha=0.6)

        # 2. Plot C+ characteristics (reflected waves heading to wall)
        for k, line in enumerate(mln.c_plus_lines):
            xs = [n.x for n in line]
            ys = [n.y for n in line]
            ax.plot(xs, ys, color="dodgerblue", lw=0.7, alpha=0.7, label="C+ waves" if k == 0 else "")
            # Mirror
            ax.plot(xs, [-y for y in ys], color="dodgerblue", lw=0.7, alpha=0.7)

        # 3. Plot Centerline
        x_max = mln.wall_nodes[-1].x
        ax.axhline(0, color="gray", linestyle="-.", lw=1.2, label="Centerline")

        # 4. Plot Wall Contours (Upper and Lower)
        wall_x = [n.x for n in mln.wall_nodes]
        wall_y = [n.y for n in mln.wall_nodes]

        # Use smooth spline for drawing wall
        x_dense = np.linspace(0.0, x_max, 250)
        y_dense = [mln.wall_spline.y(x) for x in x_dense]
        ax.plot(x_dense, y_dense, "k-", lw=2.5, label="Designed Wall Contour")
        ax.plot(x_dense, [-y for y in y_dense], "k-", lw=2.5)

        # Highlight wall nodes
        ax.scatter(wall_x, wall_y, color="black", s=15, zorder=5)
        ax.scatter(wall_x, [-y for y in wall_y], color="black", s=15, zorder=5)

        ax.set_title(
            f"Anderson 2D Planar Minimum Length Nozzle Design ($M_D = {mln.M_design:.2f}$, $N = {mln.num_rays}$)\n"
            f"Exit Area Ratio: $A/A^* = {mln.area_ratio_numerical:.4f}$ (Exact 1D: ${mln.area_ratio_exact:.4f}$, Error: ${mln.error_percent:.3f}\\%$)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Axial Coordinate, $x / y_t$", fontsize=11)
        ax.set_ylabel("Vertical Coordinate, $y / y_t$", fontsize=11)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[Visualization] MLN plot saved to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_flowfield(
        self,
        sol: MoCSolution,
        property_name: str = "M",
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Renders full symmetric color contour flowfield (Mach or Pressure)
        using triangular interpolation over discrete characteristic nodes.
        """
        x_pts = []
        y_pts = []
        vals = []

        for node in sol.all_nodes:
            if node.x <= sol.wall.x_end + 1e-4:
                x_pts.append(node.x)
                y_pts.append(node.y)
                if property_name == "M":
                    vals.append(node.M)
                elif property_name == "p":
                    vals.append(sol.gas.get_p_ratio(node.M) * sol.thrust["p0_bar"])
                elif property_name == "theta":
                    vals.append(node.theta_deg)
                else:
                    vals.append(node.M)

        # Mirror about y=0 for full nozzle visualization
        x_full = np.array(x_pts + x_pts)
        y_full = np.array(y_pts + [-y for y in y_pts])
        if property_name == "theta":
            val_full = np.array(vals + [-v for v in vals])
        else:
            val_full = np.array(vals + vals)

        fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

        # Triangular mesh interpolation
        triang = tri.Triangulation(x_full, y_full)

        cmap = "turbo" if property_name == "M" else "viridis"
        label = "Mach Number" if property_name == "M" else "Static Pressure (bar)"

        cf = ax.tricontourf(triang, val_full, levels=45, cmap=cmap)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(label, fontsize=11)

        # Characteristic lines overlay
        for line in sol.c_plus_lines:
            xs = [n.x for n in line if n.x <= sol.wall.x_end]
            ys = [n.y for n in line if n.x <= sol.wall.x_end]
            if len(xs) > 1:
                ax.plot(xs, ys, color="white", lw=0.35, alpha=0.4)
                ax.plot(xs, [-y for y in ys], color="white", lw=0.35, alpha=0.4)

        # Nozzle walls
        x_dense = np.linspace(sol.wall.x_start, sol.wall.x_end, 200)
        y_dense = [sol.wall.y(x) for x in x_dense]
        ax.plot(x_dense, y_dense, "k-", lw=3.0, label="Nozzle Wall")
        ax.plot(x_dense, [-y for y in y_dense], "k-", lw=3.0)
        ax.axhline(0, color="white", linestyle="-.", lw=1.0, alpha=0.8)

        ax.set_title(
            f"2D Planar Method of Characteristics Flowfield: {label}\n"
            f"Total Thrust: {sol.thrust['total_thrust_N']:.2f} N/m | $C_T$: {sol.thrust['thrust_coefficient_CT']:.4f} | Exit Mach: {sol.thrust['average_exit_Mach']:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Axial Position, $x$ (m)", fontsize=11)
        ax.set_ylabel("Vertical Position, $y$ (m)", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[Visualization] Flowfield plot saved to: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

