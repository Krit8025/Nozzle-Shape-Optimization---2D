"""
Thermodynamic and Gas Dynamics Engine for 2D Method of Characteristics (MoC).
Based on John D. Anderson, Modern Compressible Flow (Chapter 11 & Isentropic Relations).

Author: Antigravity MoC Clean Solver
"""

import math
import numpy as np
from scipy.optimize import brentq


class GasDynamics:
    """
    Handles 1D and 2D isentropic supersonic gas dynamics calculations.
    Default gas: Caloric perfect diatomic gas (gamma = 1.4, R = 287.05 J/(kg*K)).
    """

    def __init__(self, gamma: float = 1.4, R: float = 287.05):
        if gamma <= 1.0:
            raise ValueError(f"Gamma must be > 1.0, got {gamma}")
        self.gamma = float(gamma)
        self.R = float(R)
        self.gp1 = self.gamma + 1.0
        self.gm1 = self.gamma - 1.0
        self.g_ratio = math.sqrt(self.gp1 / self.gm1)
        self.g_inv_ratio = math.sqrt(self.gm1 / self.gp1)

        # Maximum physical Prandtl-Meyer expansion angle (M -> infinity)
        self.nu_max = (math.pi / 2.0) * (self.g_ratio - 1.0)

    # --------------------------------------------------------------------------
    # Mach Angle and Prandtl-Meyer Functions
    # --------------------------------------------------------------------------

    def get_mu(self, M: float) -> float:
        """
        Mach angle mu = arcsin(1 / M) in radians.
        Valid for M >= 1.0. For M == 1.0, mu = pi/2 (90 deg).
        """
        if M < 1.0:
            raise ValueError(f"Mach angle undefined for subsonic Mach M = {M:.4f} < 1.0")
        if M == 1.0:
            return math.pi / 2.0
        return math.asin(1.0 / M)

    def get_nu(self, M: float) -> float:
        """
        Prandtl-Meyer angle nu(M) in radians (Anderson Eq. 11.23 & 9.41).
        nu(M) = sqrt((gamma+1)/(gamma-1)) * atan(sqrt((gamma-1)/(gamma+1)*(M^2-1))) - atan(sqrt(M^2-1))
        """
        if M <= 1.0:
            return 0.0

        M2_minus_1 = M * M - 1.0
        sqrt_term = math.sqrt(M2_minus_1)
        term1 = self.g_ratio * math.atan(self.g_inv_ratio * sqrt_term)
        term2 = math.atan(sqrt_term)
        return term1 - term2

    def get_dnu_dM(self, M: float) -> float:
        """
        Analytical derivative dnu / dM:
        dnu/dM = sqrt(M^2 - 1) / (M * (1 + 0.5 * (gamma - 1) * M^2))
        """
        if M <= 1.0:
            return 0.0
        num = math.sqrt(M * M - 1.0)
        den = M * (1.0 + 0.5 * self.gm1 * M * M)
        return num / den

    def get_M_from_nu(self, target_nu: float, tolerance: float = 1e-12) -> float:
        """
        Robust inverse Prandtl-Meyer function M(nu) using bracketed root finding (brentq).
        Guaranteed to converge for any physical nu in [0, nu_max).
        """
        if target_nu <= 0.0:
            return 1.0
        if target_nu >= self.nu_max:
            raise ValueError(
                f"Requested nu = {math.degrees(target_nu):.2f}° exceeds theoretical maximum "
                f"nu_max = {math.degrees(self.nu_max):.2f}° for gamma = {self.gamma}"
            )

        # Initial asymptotic guess for bracket sizing
        # Near M=1: nu ~ (2 / (3 * (gamma+1))) * (M^2 - 1)^(3/2)
        # -> M ~ sqrt(1 + (1.5 * (gamma+1) * nu)^(2/3))
        m_guess = math.sqrt(1.0 + (1.5 * self.gp1 * target_nu) ** (2.0 / 3.0))

        # Define root equation
        def f(m):
            return self.get_nu(m) - target_nu

        # Establish bracket
        m_low = 1.0
        m_high = max(2.0, m_guess * 2.0)
        while f(m_high) < 0.0:
            m_high *= 2.0

        # Brent's method is rock-solid and machine-precision accurate
        M_sol = brentq(f, m_low, m_high, xtol=tolerance, rtol=tolerance)
        return float(M_sol)

    # --------------------------------------------------------------------------
    # 1D Isentropic Relations (Anderson Chapter 3)
    # --------------------------------------------------------------------------

    def get_T_ratio(self, M: float) -> float:
        """Static to total temperature ratio T / T0 = (1 + 0.5 * (gamma-1) * M^2)^(-1)"""
        return 1.0 / (1.0 + 0.5 * self.gm1 * M * M)

    def get_p_ratio(self, M: float) -> float:
        """Static to total pressure ratio p / p0 = (T / T0)^(gamma / (gamma-1))"""
        return self.get_T_ratio(M) ** (self.gamma / self.gm1)

    def get_rho_ratio(self, M: float) -> float:
        """Static to total density ratio rho / rho0 = (T / T0)^(1 / (gamma-1))"""
        return self.get_T_ratio(M) ** (1.0 / self.gm1)

    def get_area_ratio(self, M: float) -> float:
        """
        Isentropic 1D Area-Mach relation (A / A*):
        (A / A*)^2 = (1 / M^2) * [(2 / (gamma+1)) * (1 + 0.5 * (gamma-1) * M^2)]^((gamma+1)/(gamma-1))
        """
        if M <= 0.0:
            raise ValueError(f"Mach number must be positive, got {M}")
        factor = (2.0 / self.gp1) * (1.0 + 0.5 * self.gm1 * M * M)
        return (1.0 / M) * (factor ** (0.5 * self.gp1 / self.gm1))

    def get_M_from_area_ratio(self, area_ratio: float, supersonic: bool = True) -> float:
        """Finds Mach number from A/A* via bracketed solve."""
        if area_ratio < 1.0:
            raise ValueError(f"Area ratio cannot be < 1.0, got {area_ratio}")
        if abs(area_ratio - 1.0) < 1e-9:
            return 1.0

        def f(m):
            return self.get_area_ratio(m) - area_ratio

        if supersonic:
            m_sol = brentq(f, 1.0, 50.0)
        else:
            m_sol = brentq(f, 1e-4, 1.0)
        return float(m_sol)

    def get_state(self, M: float, p0: float, T0: float):
        """
        Computes all thermodynamic flow properties for a given Mach number and stagnation conditions.
        Returns dictionary with {p, T, rho, a, V}.
        """
        t_ratio = self.get_T_ratio(M)
        p_ratio = self.get_p_ratio(M)
        rho_ratio = self.get_rho_ratio(M)

        T = T0 * t_ratio
        p = p0 * p_ratio
        rho0 = p0 / (self.R * T0)
        rho = rho0 * rho_ratio
        a = math.sqrt(self.gamma * self.R * T)
        V = M * a

        return {
            "M": M,
            "T": T,
            "p": p,
            "rho": rho,
            "a": a,
            "V": V,
            "T_ratio": t_ratio,
            "p_ratio": p_ratio,
        }

