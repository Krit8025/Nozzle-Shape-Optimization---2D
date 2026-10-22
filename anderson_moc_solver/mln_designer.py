"""
Minimum-Length Supersonic Nozzle (MLN) Synthesizer based on 2D Planar MoC.
Reference: John D. Anderson, Modern Compressible Flow, Chapter 11, Section 11.5.

Designs a shock-free, minimum-length supersonic nozzle that produces uniform,
parallel axial flow (theta = 0, M = M_design) at the nozzle exit.
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from thermo import GasDynamics
from geometry import SplineWall
from unit_processes import (
    MoCNode,
    unit_process_interior,
    unit_process_centerline,
    unit_process_wall_cancellation,
)


class MinimumLengthNozzle:
    """
    Synthesizes and analyzes a 2D planar Minimum Length Nozzle (MLN).
    """

    def __init__(self, gas: Optional[GasDynamics] = None):
        self.gas = gas if gas is not None else GasDynamics(gamma=1.4)
        self.is_designed = False

        # Results storage
        self.M_design: float = 0.0
        self.num_rays: int = 0
        self.y_throat: float = 1.0
        self.theta_max: float = 0.0
        self.d_theta: float = 0.0

        self.throat_nodes: List[MoCNode] = []
        self.kernel_rays: List[List[MoCNode]] = []
        self.wall_nodes: List[MoCNode] = []
        self.all_nodes: List[MoCNode] = []
        self.c_plus_lines: List[List[MoCNode]] = []
        self.c_minus_lines: List[List[MoCNode]] = []
        self.wall_spline: Optional[SplineWall] = None

        self.y_exit_numerical: float = 0.0
        self.y_exit_exact: float = 0.0
        self.area_ratio_numerical: float = 0.0
        self.area_ratio_exact: float = 0.0
        self.error_percent: float = 0.0

    def design(
        self,
        M_design: float = 2.4,
        num_rays: int = 20,
        y_throat: float = 1.0,
    ) -> Dict:
        """
        Executes the Anderson Section 11.5 nozzle design algorithm.
        
        Parameters:
            M_design: Target uniform exit Mach number (e.g. 2.4)
            num_rays: Number of characteristic rays in the expansion fan
            y_throat: Half-height of the throat (default 1.0)
            
        Returns:
            Dictionary with design summary and performance metrics.
        """
        if M_design <= 1.0:
            raise ValueError(f"Design Mach must be > 1.0, got {M_design}")
        if num_rays < 2:
            raise ValueError(f"Number of rays must be >= 2, got {num_rays}")

        self.M_design = float(M_design)
        self.num_rays = int(num_rays)
        self.y_throat = float(y_throat)

        # 1. Throat Centered Expansion Fan Parameters
        nu_design = self.gas.get_nu(self.M_design)
        self.theta_max = nu_design / 2.0
        self.d_theta = self.theta_max / self.num_rays

        self.throat_nodes = []
        for i in range(self.num_rays):
            th = (i + 1) * self.d_theta
            nu = th  # Prandtl-Meyer centered expansion from sonic
            M = self.gas.get_M_from_nu(nu)
            mu = self.gas.get_mu(M)
            node = MoCNode(
                0.0,
                self.y_throat,
                M,
                th,
                nu,
                mu,
                node_type="throat",
                node_id=f"T_{i+1}",
            )
            self.throat_nodes.append(node)

        # 2. Build Expansion Kernel (Down to Centerline)
        self.kernel_rays = []
        for i in range(self.num_rays):
            ray = []
            for j in range(i):
                p_minus = self.throat_nodes[i] if j == 0 else ray[j - 1]
                p_plus = self.kernel_rays[i - 1][j]
                node_int = unit_process_interior(
                    p_minus,
                    p_plus,
                    self.gas,
                    node_id=f"K_{i+1}_{j+1}",
                )
                ray.append(node_int)

            p_minus = self.throat_nodes[0] if i == 0 else ray[-1]
            node_cl = unit_process_centerline(
                p_minus,
                self.gas,
                node_id=f"CL_{i+1}",
            )
            ray.append(node_cl)
            self.kernel_rays.append(ray)

        # 3. Build Cancellation Wall Contour (Unit Process 3)
        throat_corner = MoCNode(
            0.0,
            self.y_throat,
            1.0,
            self.theta_max,
            0.0,
            self.gas.get_mu(1.0),
            node_type="throat_corner",
            node_id="W_0",
        )
        self.wall_nodes = [throat_corner]
        prev_wall = throat_corner

        for k in range(self.num_rays):
            # The incoming C+ characteristic leaves the kernel at the outer ray
            int_node = self.kernel_rays[self.num_rays - 1][k]
            node_wall = unit_process_wall_cancellation(
                int_node,
                prev_wall,
                self.gas,
                node_id=f"W_{k+1}",
            )
            self.wall_nodes.append(node_wall)
            prev_wall = node_wall

        # 4. Extract Characteristic Line Networks for Visualization
        self._extract_characteristic_lines()

        # 5. Build Wall Spline for Continuous Representation
        wall_x = [n.x for n in self.wall_nodes]
        wall_y = [n.y for n in self.wall_nodes]
        self.wall_spline = SplineWall(wall_x, wall_y)

        # 6. Golden Reference Validation against Exact 1D Isentropic Theory
        self.y_exit_numerical = self.wall_nodes[-1].y
        self.area_ratio_numerical = self.y_exit_numerical / self.y_throat
        self.area_ratio_exact = self.gas.get_area_ratio(self.M_design)
        self.y_exit_exact = self.y_throat * self.area_ratio_exact
        self.error_percent = (
            abs(self.area_ratio_numerical - self.area_ratio_exact)
            / self.area_ratio_exact
            * 100.0
        )

        self.is_designed = True

        return {
            "M_design": self.M_design,
            "num_rays": self.num_rays,
            "theta_max_deg": math.degrees(self.theta_max),
            "L_nozzle": self.wall_nodes[-1].x,
            "y_exit_numerical": self.y_exit_numerical,
            "y_exit_exact": self.y_exit_exact,
            "area_ratio_numerical": self.area_ratio_numerical,
            "area_ratio_exact": self.area_ratio_exact,
            "error_percent": self.error_percent,
            "total_nodes": len(self.all_nodes),
        }

    def _extract_characteristic_lines(self):
        """Builds organized lists of C+ and C- lines for accurate plotting."""
        self.c_plus_lines = []
        self.c_minus_lines = []
        self.all_nodes = []

        # All C- lines (each ray in kernel)
        for i, ray in enumerate(self.kernel_rays):
            c_minus = [self.throat_nodes[i]] + ray
            self.c_minus_lines.append(c_minus)
            for n in ray:
                if n not in self.all_nodes:
                    self.all_nodes.append(n)

        # All C+ lines (from centerline through kernel to wall)
        for k in range(self.num_rays):
            line_k = []
            for i in range(k, self.num_rays):
                line_k.append(self.kernel_rays[i][k])
            # Add corresponding wall node
            line_k.append(self.wall_nodes[k + 1])
            self.c_plus_lines.append(line_k)

        for w in self.wall_nodes:
            if w not in self.all_nodes:
                self.all_nodes.append(w)

