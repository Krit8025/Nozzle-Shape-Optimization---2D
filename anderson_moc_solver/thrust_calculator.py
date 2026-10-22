"""
Thrust and Aerodynamic Performance Engine for 2D Planar Supersonic Nozzles.
Calculates mass flow rate, momentum flux, pressure thrust, and thrust coefficient C_T.
Reference: Fernandes et al. (2023) Eqs. 17-22 & Sutton/Biblarz Rocket Propulsion Elements.
"""

import math
from typing import List, Dict, Optional
import numpy as np

from thermo import GasDynamics
from unit_processes import MoCNode


class ThrustCalculator:
    """
    Computes performance metrics from exit-plane flow profiles or characteristic cut-planes.
    """

    def __init__(
        self,
        gas: Optional[GasDynamics] = None,
        p0: float = 1.0e6,      # Chamber stagnation pressure (Pa)
        T0: float = 3000.0,     # Chamber stagnation temperature (K)
        p_amb: float = 101325.0 # Ambient pressure (Pa)
    ):
        self.gas = gas if gas is not None else GasDynamics(gamma=1.4)
        self.p0 = float(p0)
        self.T0 = float(T0)
        self.p_amb = float(p_amb)

    def compute_from_nodes(
        self,
        exit_nodes: List[MoCNode],
        y_throat: float = 1.0,
    ) -> Dict:
        """
        Integrates exit properties across exit nodes from centerline (y=0) to wall (y=y_exit).
        
        Parameters:
            exit_nodes: List of MoCNode objects across the exit cross-section
            y_throat: Half-height of the throat for normalization
            
        Returns:
            Dictionary with complete aerodynamic performance metrics.
        """
        if len(exit_nodes) < 2:
            raise ValueError(f"Need at least 2 exit nodes to integrate, got {len(exit_nodes)}")

        # Extract sorted profiles from y=0 to y=y_exit
        y_list = [n.y for n in exit_nodes]
        order = np.argsort(y_list)

        y_arr = np.array([exit_nodes[i].y for i in order])
        M_arr = np.array([exit_nodes[i].M for i in order])
        th_arr = np.array([exit_nodes[i].theta for i in order])

        y_exit = float(y_arr[-1])
        area_ratio = y_exit / y_throat

        # Thermodynamics at each exit point
        t_ratio = 1.0 / (1.0 + 0.5 * self.gas.gm1 * M_arr**2)
        p_ratio = t_ratio ** (self.gas.gamma / self.gas.gm1)

        T_arr = self.T0 * t_ratio
        p_arr = self.p0 * p_ratio
        rho_arr = p_arr / (self.gas.R * T_arr)
        a_arr = np.sqrt(self.gas.gamma * self.gas.R * T_arr)
        V_arr = M_arr * a_arr
        Vx_arr = V_arr * np.cos(th_arr)
        Vy_arr = V_arr * np.sin(th_arr)

        # Integrals per unit depth (multiply half-nozzle integral by 2 for symmetric full nozzle)
        # 1. Mass flow rate: m_dot = 2 * integral(rho * Vx * dy)
        mdot_integrand = rho_arr * Vx_arr
        mdot_half = float(np.trapezoid(mdot_integrand, y_arr))
        mdot_total = 2.0 * mdot_half

        # 2. Momentum thrust: F_mom = 2 * integral(rho * Vx^2 * dy)
        mom_integrand = rho_arr * (Vx_arr**2)
        F_mom_half = float(np.trapezoid(mom_integrand, y_arr))
        F_mom_total = 2.0 * F_mom_half

        # 3. Pressure thrust: F_press = 2 * integral((p - p_amb) * dy)
        press_integrand = p_arr - self.p_amb
        F_press_half = float(np.trapezoid(press_integrand, y_arr))
        F_press_total = 2.0 * F_press_half

        # 4. Total thrust: F = F_mom + F_press
        F_total = F_mom_total + F_press_total

        # 5. Thrust coefficient: C_T = F / (p0 * A_throat)
        # Throat area for 2D planar unit width = 2 * y_throat * 1.0
        A_throat = 2.0 * y_throat
        A_exit = 2.0 * y_exit
        C_T = F_total / (self.p0 * A_throat)

        # 6. Ideal 1D Isentropic Comparison
        # Choked mass flow rate
        mdot_choked = (
            A_throat
            * self.p0
            * math.sqrt(self.gas.gamma / (self.gas.R * self.T0))
            * (2.0 / self.gas.gp1) ** (0.5 * self.gas.gp1 / self.gas.gm1)
        )

        # Ideal 1D exit Mach from area ratio
        try:
            M_1d = self.gas.get_M_from_area_ratio(area_ratio, supersonic=True)
            state_1d = self.gas.get_state(M_1d, self.p0, self.T0)
            F_ideal = mdot_choked * state_1d["V"] + (state_1d["p"] - self.p_amb) * A_exit
            C_T_ideal = F_ideal / (self.p0 * A_throat)
            divergence_efficiency = F_total / F_ideal if F_ideal > 0 else 1.0
        except Exception:
            M_1d = float(np.mean(M_arr))
            F_ideal = F_total
            C_T_ideal = C_T
            divergence_efficiency = 1.0

        return {
            "p0_bar": self.p0 / 1.0e5,
            "T0_K": self.T0,
            "p_amb_bar": self.p_amb / 1.0e5,
            "y_throat": y_throat,
            "y_exit": y_exit,
            "area_ratio": area_ratio,
            "mass_flow_rate_kg_s": mdot_total,
            "mass_flow_choked_kg_s": mdot_choked,
            "momentum_thrust_N": F_mom_total,
            "pressure_thrust_N": F_press_total,
            "total_thrust_N": F_total,
            "thrust_coefficient_CT": C_T,
            "ideal_thrust_N": F_ideal,
            "ideal_thrust_coefficient_CT": C_T_ideal,
            "divergence_efficiency": divergence_efficiency,
            "average_exit_Mach": float(np.mean(M_arr)),
            "average_exit_pressure_Pa": float(np.mean(p_arr)),
            "max_divergence_angle_deg": float(np.degrees(np.max(np.abs(th_arr)))),
        }

