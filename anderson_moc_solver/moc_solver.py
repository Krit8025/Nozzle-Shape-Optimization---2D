"""
General 2D Planar Method of Characteristics (MoC) Solver.
Analyzes supersonic flow through any given diffusing / divergent nozzle geometry.
Reference: John D. Anderson, Modern Compressible Flow (Chapter 11) & Fernandes et al. (2023).
"""

import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from thermo import GasDynamics
from geometry import NozzleWall, SplineWall, BezierWall, ParabolicWall
from unit_processes import (
    MoCNode,
    unit_process_interior,
    unit_process_centerline,
    unit_process_wall_reflection,
)
from thrust_calculator import ThrustCalculator


class MoCSolution:
    """Stores full flowfield results from a 2D planar MoC simulation."""

    def __init__(
        self,
        wall: NozzleWall,
        gas: GasDynamics,
        all_nodes: List[MoCNode],
        c_plus_lines: List[List[MoCNode]],
        c_minus_lines: List[List[MoCNode]],
        wall_nodes: List[MoCNode],
        centerline_nodes: List[MoCNode],
        exit_nodes: List[MoCNode],
        thrust_metrics: Dict,
    ):
        self.wall = wall
        self.gas = gas
        self.all_nodes = all_nodes
        self.c_plus_lines = c_plus_lines
        self.c_minus_lines = c_minus_lines
        self.wall_nodes = wall_nodes
        self.centerline_nodes = centerline_nodes
        self.exit_nodes = exit_nodes
        self.thrust = thrust_metrics

    def summary(self) -> str:
        """Returns a formatted performance and convergence summary."""
        lines = [
            "===========================================================",
            "             2D PLANAR MOC NOZZLE SIMULATION               ",
            "===========================================================",
            f"Nozzle Length (L)       : {self.wall.x_end - self.wall.x_start:.4f} m",
            f"Throat Half-Height (yt) : {self.wall.y_throat:.4f} m",
            f"Exit Half-Height (ye)   : {self.wall.y_exit:.4f} m",
            f"Geometric Area Ratio    : {self.wall.y_exit / self.wall.y_throat:.4f}",
            "-----------------------------------------------------------",
            f"Chamber Pressure (p0)   : {self.thrust['p0_bar']:.2f} bar",
            f"Chamber Temp (T0)       : {self.thrust['T0_K']:.1f} K",
            f"Ambient Pressure (pamb) : {self.thrust['p_amb_bar']:.3f} bar",
            "-----------------------------------------------------------",
            f"Total Nodes Computed    : {len(self.all_nodes)}",
            f"Average Exit Mach       : {self.thrust['average_exit_Mach']:.3f}",
            f"Mass Flow Rate (mdot)   : {self.thrust['mass_flow_rate_kg_s']:.3f} kg/s",
            f"Total Planar Thrust (F) : {self.thrust['total_thrust_N']:.2f} N/m",
            f"Thrust Coefficient (CT) : {self.thrust['thrust_coefficient_CT']:.4f}",
            f"Ideal 1D Thrust (F_1d)  : {self.thrust['ideal_thrust_N']:.2f} N/m",
            f"Ideal 1D CT (CT_1d)     : {self.thrust['ideal_thrust_coefficient_CT']:.4f}",
            f"Nozzle Efficiency (eta) : {self.thrust['divergence_efficiency']*100.0:.2f} %",
            "===========================================================",
        ]
        return "\n".join(lines)


class GeneralMoCSolver:
    """
    2D Method of Characteristics solver for any given diffusing nozzle contour.
    """

    def __init__(self, gas: Optional[GasDynamics] = None):
        self.gas = gas if gas is not None else GasDynamics(gamma=1.4)

    def solve(
        self,
        wall: NozzleWall,
        num_rays: int = 25,
        p0: float = 1.0e6,
        T0: float = 3000.0,
        p_amb: float = 101325.0,
        max_downstream_lines: int = 150,
    ) -> MoCSolution:
        """
        Solves the supersonic flowfield for the given nozzle geometry.
        
        Parameters:
            wall: NozzleWall instance defining y_w(x)
            num_rays: Number of initial expansion rays at throat
            p0: Stagnation pressure (Pa)
            T0: Stagnation temperature (K)
            p_amb: Ambient pressure (Pa)
            max_downstream_lines: Maximum characteristic lines to step downstream
            
        Returns:
            MoCSolution instance
        """
        yt = wall.y_throat
        theta_throat = wall.theta(wall.x_start)
        # Ensure a small positive turning angle at throat for expansion
        theta_exp = max(math.radians(1.5), theta_throat)
        d_theta = theta_exp / num_rays

        # 1. Throat Centered Expansion Fan (Sonic source corner)
        throat_nodes = []
        for i in range(num_rays):
            th = (i + 1) * d_theta
            nu = th
            M = self.gas.get_M_from_nu(nu)
            mu = self.gas.get_mu(M)
            node = MoCNode(
                wall.x_start,
                yt,
                M,
                th,
                nu,
                mu,
                node_type="throat",
                node_id=f"T_{i+1}",
            )
            throat_nodes.append(node)

        # 2. Expansion Kernel Calculation
        kernel_rays: List[List[MoCNode]] = []
        all_nodes: List[MoCNode] = list(throat_nodes)
        centerline_nodes: List[MoCNode] = []

        for i in range(num_rays):
            ray = []
            for j in range(i):
                p_minus = throat_nodes[i] if j == 0 else ray[j - 1]
                p_plus = kernel_rays[i - 1][j]
                node_int = unit_process_interior(
                    p_minus,
                    p_plus,
                    self.gas,
                    node_id=f"K_{i+1}_{j+1}",
                )
                ray.append(node_int)
                all_nodes.append(node_int)

            p_minus = throat_nodes[0] if i == 0 else ray[-1]
            node_cl = unit_process_centerline(
                p_minus,
                self.gas,
                node_id=f"CL_{i+1}",
            )
            ray.append(node_cl)
            centerline_nodes.append(node_cl)
            all_nodes.append(node_cl)
            kernel_rays.append(ray)

        # 3. Form Initial Left-Running (C+) Characteristics
        # Emerging from the kernel and extending up to the solid wall
        c_plus_lines: List[List[MoCNode]] = []
        wall_nodes: List[MoCNode] = []

        for j in range(num_rays):
            line = []
            for i in range(j, num_rays):
                line.append(kernel_rays[i][j])
            # Extend to wall with reflection (Unit Process 4)
            node_w = unit_process_wall_reflection(
                line[-1],
                wall,
                self.gas,
                node_id=f"W_init_{j+1}",
            )
            line.append(node_w)
            wall_nodes.append(node_w)
            all_nodes.append(node_w)
            c_plus_lines.append(line)

        # 4. March Downstream through the Diffusing Duct (if needed)
        # If the last characteristic hasn't reached the nozzle exit wall.x_end,
        # continue stepping downstream with wall and centerline reflections.
        step = 0
        while c_plus_lines[-1][-1].x < wall.x_end and step < max_downstream_lines:
            step += 1
            prev_line = c_plus_lines[-1]
            if len(prev_line) < 3:
                break

            curr_line = []
            # A. Centerline point from first interior point of previous line
            node_cl = unit_process_centerline(
                prev_line[1],
                self.gas,
                node_id=f"CL_m_{step}",
            )
            curr_line.append(node_cl)
            centerline_nodes.append(node_cl)
            all_nodes.append(node_cl)

            # B. Interior points
            for k in range(1, len(prev_line) - 1):
                node_int = unit_process_interior(
                    prev_line[k + 1],
                    curr_line[k - 1],
                    self.gas,
                    node_id=f"INT_m_{step}_{k}",
                )
                curr_line.append(node_int)
                all_nodes.append(node_int)

            # C. Wall reflection
            node_w = unit_process_wall_reflection(
                curr_line[-1],
                wall,
                self.gas,
                node_id=f"W_m_{step}",
            )
            curr_line.append(node_w)
            wall_nodes.append(node_w)
            all_nodes.append(node_w)

            c_plus_lines.append(curr_line)

        # 5. Extract C- lines for full grid visualization
        c_minus_lines = self._build_c_minus_lines(kernel_rays, throat_nodes, c_plus_lines)

        # 6. Extract Exit Plane Profile (Virtual Characteristic Interpolation)
        exit_nodes = self._extract_exit_profile(c_plus_lines, wall.x_end, wall)

        # 7. Compute Thrust and Aerodynamic Performance
        calculator = ThrustCalculator(gas=self.gas, p0=p0, T0=T0, p_amb=p_amb)
        thrust_metrics = calculator.compute_from_nodes(exit_nodes, y_throat=yt)

        return MoCSolution(
            wall=wall,
            gas=self.gas,
            all_nodes=all_nodes,
            c_plus_lines=c_plus_lines,
            c_minus_lines=c_minus_lines,
            wall_nodes=wall_nodes,
            centerline_nodes=centerline_nodes,
            exit_nodes=exit_nodes,
            thrust_metrics=thrust_metrics,
        )

    def _build_c_minus_lines(
        self,
        kernel_rays: List[List[MoCNode]],
        throat_nodes: List[MoCNode],
        c_plus_lines: List[List[MoCNode]],
    ) -> List[List[MoCNode]]:
        """Reconstructs right-running C- lines for complete grid rendering."""
        c_minus_lines = []
        for i, ray in enumerate(kernel_rays):
            line = [throat_nodes[i]] + ray
            c_minus_lines.append(line)
        return c_minus_lines

    def _extract_exit_profile(
        self,
        c_plus_lines: List[List[MoCNode]],
        x_exit: float,
        wall: NozzleWall,
    ) -> List[MoCNode]:
        """
        Finds or interpolates flow properties across the nozzle exit cross-section.
        Uses Fernandes et al. (2023) Eq. 21 virtual characteristic interpolation.
        """
        # Find the line just before x_exit and the line just after x_exit
        line_before = None
        line_after = None

        for line in c_plus_lines:
            wall_pt = line[-1]
            if wall_pt.x <= x_exit:
                line_before = line
            else:
                line_after = line
                break

        if line_before is None:
            # All lines beyond exit; take first line
            return c_plus_lines[0]
        if line_after is None:
            # Exit is past all computed lines; take the final line
            return c_plus_lines[-1]

        # Interpolate between line_before and line_after at exit fraction
        xb = line_before[-1].x
        xa = line_after[-1].x
        frac = (x_exit - xb) / (xa - xb) if abs(xa - xb) > 1e-6 else 0.5
        frac = max(0.0, min(1.0, frac))

        n_pts = min(len(line_before), len(line_after))
        exit_nodes = []

        for i in range(n_pts):
            nb = line_before[i]
            na = line_after[i]
            y_interp = nb.y + frac * (na.y - nb.y)
            M_interp = nb.M + frac * (na.M - nb.M)
            th_interp = nb.theta + frac * (na.theta - nb.theta)
            nu_interp = nb.nu + frac * (na.nu - nb.nu)
            mu_interp = self.gas.get_mu(M_interp)

            node_exit = MoCNode(
                x_exit,
                y_interp,
                M_interp,
                th_interp,
                nu_interp,
                mu_interp,
                node_type="exit",
                node_id=f"EX_{i+1}",
            )
            exit_nodes.append(node_exit)

        # Ensure top exit node matches wall exit
        exit_nodes[-1].y = wall.y(x_exit)
        exit_nodes[-1].theta = wall.theta(x_exit)
        exit_nodes[0].y = 0.0
        exit_nodes[0].theta = 0.0

        return exit_nodes

