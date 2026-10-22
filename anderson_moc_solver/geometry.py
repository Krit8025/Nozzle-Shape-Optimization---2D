"""
Nozzle Wall Geometry Module for 2D Planar Method of Characteristics.
Supports:
  1. Cubic Bezier Wall (control-point parameterized)
  2. Spline Wall (interpolated from discrete points, e.g. from MLN design)
  3. Parabolic Bell Wall (Rao-type approximation)
  4. Linear / Conical Wall

Provides analytical / spline derivatives and robust characteristic intersection solvers.
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


class NozzleWall(ABC):
    """Abstract base class for nozzle contour geometries."""

    @abstractmethod
    def y(self, x: float) -> float:
        """Wall ordinate y at axial position x."""
        pass

    @abstractmethod
    def dydx(self, x: float) -> float:
        """Slope dy/dx at axial position x."""
        pass

    def theta(self, x: float) -> float:
        """Wall inclination angle in radians at axial position x."""
        return math.atan(self.dydx(x))

    @property
    @abstractmethod
    def x_start(self) -> float:
        """Axial position of nozzle throat."""
        pass

    @property
    @abstractmethod
    def x_end(self) -> float:
        """Axial position of nozzle exit."""
        pass

    @property
    @abstractmethod
    def y_throat(self) -> float:
        """Throat half-height."""
        pass

    @property
    @abstractmethod
    def y_exit(self) -> float:
        """Exit half-height."""
        pass

    def intersect_characteristic(
        self,
        x_up: float,
        y_up: float,
        slope: float,
        x_max: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Computes the intersection (x_w, y_w) of a straight characteristic ray
        passing through (x_up, y_up) with slope `slope` and this wall contour y_w(x).
        Ray equation: y_line(x) = y_up + slope * (x - x_up).
        """
        if x_max is None:
            x_max = self.x_end * 1.5  # allow slight margin beyond exit

        def residual(x_val):
            # If beyond domain, extrapolate linearly using endpoint derivative
            if x_val > self.x_end:
                y_w = self.y_exit + self.dydx(self.x_end) * (x_val - self.x_end)
            elif x_val < self.x_start:
                y_w = self.y_throat + self.dydx(self.x_start) * (x_val - self.x_start)
            else:
                y_w = self.y(x_val)
            y_line = y_up + slope * (x_val - x_up)
            return y_w - y_line

        x_left = max(self.x_start, x_up)
        r_left = residual(x_left)

        # If already at or above wall (within numerical tol)
        if abs(r_left) < 1e-11:
            return x_left, y_up

        # Search for right bracket where residual flips sign
        x_right = x_left + max(0.01, (self.x_end - self.x_start) * 0.05)
        step = max(0.02, (self.x_end - self.x_start) * 0.1)

        while x_right < x_max:
            if residual(x_right) * r_left <= 0.0:
                break
            x_right += step

        if residual(x_right) * r_left > 0.0:
            # Did not cross within x_max; clamp to exit or extrapolation
            x_sol = (self.y_exit - y_up + slope * x_up) / slope if abs(slope) > 1e-6 else self.x_end
            return x_sol, y_up + slope * (x_sol - x_up)

        x_sol = brentq(residual, x_left, x_right, xtol=1e-10)
        y_sol = y_up + slope * (x_sol - x_up)
        return float(x_sol), float(y_sol)


class BezierWall(NozzleWall):
    """
    Parametric Cubic Bezier Wall defined by 4 control points P0, P1, P2, P3.
    P0 = (x0, y0): Throat start
    P1 = (x1, y1): First control point
    P2 = (x2, y2): Second control point
    P3 = (x3, y3): Exit point
    """

    def __init__(
        self,
        P0: Tuple[float, float],
        P1: Tuple[float, float],
        P2: Tuple[float, float],
        P3: Tuple[float, float],
    ):
        self.P0 = np.array(P0, dtype=float)
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        self.P3 = np.array(P3, dtype=float)

    @property
    def x_start(self) -> float:
        return self.P0[0]

    @property
    def x_end(self) -> float:
        return self.P3[0]

    @property
    def y_throat(self) -> float:
        return self.P0[1]

    @property
    def y_exit(self) -> float:
        return self.P3[1]

    def get_point(self, t: float) -> Tuple[float, float]:
        """Calculates (x, y) for parameter t in [0, 1]."""
        t = np.clip(t, 0.0, 1.0)
        omt = 1.0 - t
        pt = (
            omt**3 * self.P0
            + 3.0 * omt**2 * t * self.P1
            + 3.0 * omt * t**2 * self.P2
            + t**3 * self.P3
        )
        return float(pt[0]), float(pt[1])

    def get_derivatives(self, t: float) -> Tuple[float, float]:
        """Calculates dx/dt, dy/dt for parameter t in [0, 1]."""
        t = np.clip(t, 0.0, 1.0)
        omt = 1.0 - t
        dpt = (
            3.0 * omt**2 * (self.P1 - self.P0)
            + 6.0 * omt * t * (self.P2 - self.P1)
            + 3.0 * t**2 * (self.P3 - self.P2)
        )
        return float(dpt[0]), float(dpt[1])

    def get_t_from_x(self, x_target: float) -> float:
        """Finds parameter t corresponding to axial station x."""
        if x_target <= self.x_start:
            return 0.0
        if x_target >= self.x_end:
            return 1.0

        def f(t_val):
            return self.get_point(t_val)[0] - x_target

        return float(brentq(f, 0.0, 1.0, xtol=1e-10))

    def y(self, x: float) -> float:
        t = self.get_t_from_x(x)
        return self.get_point(t)[1]

    def dydx(self, x: float) -> float:
        t = self.get_t_from_x(x)
        dxdt, dydt = self.get_derivatives(t)
        if abs(dxdt) < 1e-12:
            return 0.0
        return dydt / dxdt


class SplineWall(NozzleWall):
    """
    Contour interpolated through discrete points (x, y) using a cubic spline.
    Ideal for representing synthesized Minimum-Length Nozzle (MLN) contours.
    """

    def __init__(self, x_coords: List[float], y_coords: List[float]):
        x_arr = np.asarray(x_coords, dtype=float)
        y_arr = np.asarray(y_coords, dtype=float)

        # Ensure monotonically increasing x
        order = np.argsort(x_arr)
        self.x_data = x_arr[order]
        self.y_data = y_arr[order]

        # Natural / clamped cubic spline
        self.spline = CubicSpline(self.x_data, self.y_data, bc_type="natural")
        self.spline_deriv = self.spline.derivative()

    @property
    def x_start(self) -> float:
        return float(self.x_data[0])

    @property
    def x_end(self) -> float:
        return float(self.x_data[-1])

    @property
    def y_throat(self) -> float:
        return float(self.y_data[0])

    @property
    def y_exit(self) -> float:
        return float(self.y_data[-1])

    def y(self, x: float) -> float:
        x_clamped = np.clip(x, self.x_start, self.x_end)
        return float(self.spline(x_clamped))

    def dydx(self, x: float) -> float:
        x_clamped = np.clip(x, self.x_start, self.x_end)
        return float(self.spline_deriv(x_clamped))


class ParabolicWall(NozzleWall):
    """
    Rao-type parabolic approximation for bell nozzle contours:
    y(x) = a * x^2 + b * x + c
    Matching:
      - x = 0: y = y_throat, dy/dx = tan(theta_i)
      - x = L: y = y_exit, dy/dx = tan(theta_e)
    """

    def __init__(
        self,
        L: float,
        y_throat: float,
        y_exit: float,
        theta_i_deg: float = 30.0,
        theta_e_deg: float = 8.0,
    ):
        self.L = float(L)
        self._yt = float(y_throat)
        self._ye = float(y_exit)
        self.theta_i = math.radians(theta_i_deg)
        self.theta_e = math.radians(theta_e_deg)

        m_i = math.tan(self.theta_i)
        m_e = math.tan(self.theta_e)

        # Quadratic polynomial y(x) = a*x^2 + b*x + c
        # c = y_throat
        # b = m_i
        # a = (y_exit - y_throat - m_i * L) / L^2
        self.c = self._yt
        self.b = m_i
        self.a = (self._ye - self._yt - m_i * self.L) / (self.L**2)

    @property
    def x_start(self) -> float:
        return 0.0

    @property
    def x_end(self) -> float:
        return self.L

    @property
    def y_throat(self) -> float:
        return self._yt

    @property
    def y_exit(self) -> float:
        return self._ye

    def y(self, x: float) -> float:
        x_clamped = np.clip(x, 0.0, self.L)
        return float(self.a * x_clamped**2 + self.b * x_clamped + self.c)

    def dydx(self, x: float) -> float:
        x_clamped = np.clip(x, 0.0, self.L)
        return float(2.0 * self.a * x_clamped + self.b)

