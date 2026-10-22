"""
Unit Processes for 2D Planar Method of Characteristics (MoC).
Reference: John D. Anderson, Modern Compressible Flow, Chapter 11, Sections 11.4 & 11.5.

All unit processes use 2nd-order modified Euler average slopes for spatial coordinates.
"""

import math
from typing import Optional, Any
from thermo import GasDynamics


class MoCNode:
    """
    Represents a discrete calculation point in the characteristic grid.
    Tracks state variables (M, theta, nu, mu, K+, K-) and physical coordinates (x, y).
    """

    def __init__(
        self,
        x: float,
        y: float,
        M: float,
        theta: float,
        nu: float,
        mu: float,
        node_type: str = "interior",
        node_id: Optional[Any] = None,
    ):
        self.x = float(x)
        self.y = float(y)
        self.M = float(M)
        self.theta = float(theta)  # radians
        self.nu = float(nu)        # radians
        self.mu = float(mu)        # radians
        self.node_type = str(node_type)
        self.node_id = node_id

        # Riemann invariants
        # K+ = theta - nu (constant along left-running C+ characteristic)
        # K- = theta + nu (constant along right-running C- characteristic)
        self.K_plus = self.theta - self.nu
        self.K_minus = self.theta + self.nu

    @property
    def theta_deg(self) -> float:
        """Flow angle in degrees."""
        return math.degrees(self.theta)

    @property
    def nu_deg(self) -> float:
        """Prandtl-Meyer angle in degrees."""
        return math.degrees(self.nu)

    @property
    def mu_deg(self) -> float:
        """Mach angle in degrees."""
        return math.degrees(self.mu)

    def __repr__(self) -> str:
        return (
            f"MoCNode({self.node_type}, id={self.node_id}: x={self.x:.4f}, y={self.y:.4f}, "
            f"M={self.M:.4f}, theta={self.theta_deg:.2f} deg, nu={self.nu_deg:.2f} deg)"
        )


def unit_process_interior(
    node_minus: MoCNode,
    node_plus: MoCNode,
    gas: GasDynamics,
    node_id: Optional[Any] = None,
) -> MoCNode:
    """
    Unit Process 1: Internal Flow Point (Anderson Section 11.4).
    Intersection of:
      - Right-running C- characteristic from node_minus (Point 1)
      - Left-running C+ characteristic from node_plus (Point 2)
    """
    # 1. State solution from compatibility equations
    # theta3 + nu3 = K_minus_1
    # theta3 - nu3 = K_plus_2
    K_m1 = node_minus.K_minus
    K_p2 = node_plus.K_plus

    theta3 = 0.5 * (K_m1 + K_p2)
    nu3 = 0.5 * (K_m1 - K_p2)

    # Physical safety
    if nu3 < 0.0:
        nu3 = 0.0

    M3 = gas.get_M_from_nu(nu3)
    mu3 = gas.get_mu(M3)

    # 2. Geometry solution via 2nd-order average slopes (Modified Euler)
    # Average slope along C- from Point 1 to Point 3
    theta_bar_13 = 0.5 * (node_minus.theta + theta3)
    mu_bar_13 = 0.5 * (node_minus.mu + mu3)
    m_minus = math.tan(theta_bar_13 - mu_bar_13)

    # Average slope along C+ from Point 2 to Point 3
    theta_bar_23 = 0.5 * (node_plus.theta + theta3)
    mu_bar_23 = 0.5 * (node_plus.mu + mu3)
    m_plus = math.tan(theta_bar_23 + mu_bar_23)

    if abs(m_plus - m_minus) < 1e-8:
        raise ValueError("Characteristics C+ and C- are nearly parallel; cannot compute intersection.")

    # Intersection of y - y1 = m_minus*(x - x1) and y - y2 = m_plus*(x - x2)
    x3 = (node_minus.y - node_plus.y + m_plus * node_plus.x - m_minus * node_minus.x) / (m_plus - m_minus)
    y3 = node_minus.y + m_minus * (x3 - node_minus.x)

    return MoCNode(x3, y3, M3, theta3, nu3, mu3, node_type="interior", node_id=node_id)


def unit_process_centerline(
    node_minus: MoCNode,
    gas: GasDynamics,
    node_id: Optional[Any] = None,
) -> MoCNode:
    """
    Unit Process 2: Centerline / Axis of Symmetry Point (Anderson Section 11.4).
    Intersection of:
      - Right-running C- characteristic from node_minus (Point 1)
      - Axis of symmetry y = 0
    Centerline boundary condition: theta3 = 0.
    """
    theta3 = 0.0
    # Along C-: theta3 + nu3 = K_minus_1 -> nu3 = K_minus_1
    nu3 = node_minus.K_minus
    if nu3 < 0.0:
        nu3 = 0.0

    M3 = gas.get_M_from_nu(nu3)
    mu3 = gas.get_mu(M3)

    # Average slope along C-
    theta_bar_13 = 0.5 * (node_minus.theta + theta3)
    mu_bar_13 = 0.5 * (node_minus.mu + mu3)
    m_minus = math.tan(theta_bar_13 - mu_bar_13)

    if abs(m_minus) < 1e-8:
        raise ValueError("Characteristic C- slope is nearly zero; cannot intersect centerline.")

    y3 = 0.0
    x3 = node_minus.x - (node_minus.y / m_minus)

    return MoCNode(x3, y3, M3, theta3, nu3, mu3, node_type="centerline", node_id=node_id)


def unit_process_wall_cancellation(
    node_plus: MoCNode,
    prev_wall_node: MoCNode,
    gas: GasDynamics,
    node_id: Optional[Any] = None,
) -> MoCNode:
    """
    Unit Process 3: Solid Wall Point for Nozzle Contour Design / Wave Cancellation.
    (Anderson Section 11.5 - Supersonic Nozzle Design).
    
    The wall contour is turned such that incoming left-running C+ characteristic
    is absorbed without reflection.
    Condition for cancellation: theta3 = theta_incoming = node_plus.theta.
    Along C+: theta3 - nu3 = theta_plus - nu_plus -> nu3 = node_plus.nu -> M3 = node_plus.M.
    """
    theta3 = node_plus.theta
    nu3 = node_plus.nu
    M3 = node_plus.M
    mu3 = node_plus.mu

    # Slope of C+ characteristic from node_plus (Point 2) to wall (Point 3)
    # Average slope along C+: 0.5 * (theta2 + mu2 + theta3 + mu3) = theta2 + mu2
    m_char = math.tan(node_plus.theta + node_plus.mu)

    # Average slope of the wall streamline from prev_wall_node (Point 4) to Point 3
    theta_bar_wall = 0.5 * (prev_wall_node.theta + theta3)
    m_wall = math.tan(theta_bar_wall)

    if abs(m_char - m_wall) < 1e-8:
        raise ValueError("Characteristic line and wall streamline are nearly parallel.")

    # Intersection of wall streamline and C+ line
    x3 = (node_plus.y - prev_wall_node.y + m_wall * prev_wall_node.x - m_char * node_plus.x) / (m_wall - m_char)
    y3 = prev_wall_node.y + m_wall * (x3 - prev_wall_node.x)

    return MoCNode(x3, y3, M3, theta3, nu3, mu3, node_type="wall_design", node_id=node_id)


def unit_process_wall_reflection(
    node_plus: MoCNode,
    wall_geometry: Any,
    gas: GasDynamics,
    node_id: Optional[Any] = None,
    max_corrector_iters: int = 3,
) -> MoCNode:
    """
    Unit Process 4: Solid Wall Point for Arbitrary Prescribed Geometry (Direct Analysis).
    (Anderson Section 11.4 & Fernandes et al. Section 3.3).
    
    An incoming left-running C+ characteristic strikes a fixed wall y = y_w(x).
    Boundary condition: flow must be tangent to the wall -> theta3 = arctan(y'_w(x3)).
    Along C+: theta3 - nu3 = K_plus_2 -> nu3 = theta3 - K_plus_2.
    A reflected right-running C- wave is generated with K_minus_3 = theta3 + nu3.
    """
    # Initial predictor slope using upstream properties
    m_char = math.tan(node_plus.theta + node_plus.mu)

    # Find first intersection with wall
    x3, y3 = wall_geometry.intersect_characteristic(node_plus.x, node_plus.y, m_char)
    theta3 = wall_geometry.theta(x3)
    nu3 = theta3 - node_plus.K_plus
    if nu3 < 0.0:
        nu3 = 0.0
    M3 = gas.get_M_from_nu(nu3)
    mu3 = gas.get_mu(M3)

    # Corrector loop (Modified Euler with average characteristic slope)
    for _ in range(max_corrector_iters):
        theta_bar = 0.5 * (node_plus.theta + theta3)
        mu_bar = 0.5 * (node_plus.mu + mu3)
        m_char_avg = math.tan(theta_bar + mu_bar)

        x3_new, y3_new = wall_geometry.intersect_characteristic(node_plus.x, node_plus.y, m_char_avg)
        theta3_new = wall_geometry.theta(x3_new)
        nu3_new = theta3_new - node_plus.K_plus
        if nu3_new < 0.0:
            nu3_new = 0.0
        M3_new = gas.get_M_from_nu(nu3_new)
        mu3_new = gas.get_mu(M3_new)

        diff = abs(x3_new - x3) + abs(y3_new - y3)
        x3, y3, theta3, nu3, M3, mu3 = x3_new, y3_new, theta3_new, nu3_new, M3_new, mu3_new
        if diff < 1e-9:
            break

    return MoCNode(x3, y3, M3, theta3, nu3, mu3, node_type="wall_reflection", node_id=node_id)
