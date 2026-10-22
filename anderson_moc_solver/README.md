# Anderson 2D Planar Method of Characteristics (MoC) Solver

A clean, modular, and error-free 2D Planar Method of Characteristics (MoC) solver and Differential Evolution shape optimizer for supersonic diffusing nozzles, formulated using **John D. Anderson's *Modern Compressible Flow*, Chapter 11** as the golden reference.

For the comprehensive project summary, theory, and detailed CFD benchmark comparisons, see the root [**`README.md`**](../README.md).

---

## File Architecture

| File | Description |
| :--- | :--- |
| [`thermo.py`](thermo.py) | High-precision gas dynamics engine: $\nu(M)$, $\mu(M)$, robust bracketed inverse $\text{inv}\_\nu(\nu)$ ($\approx 10^{-13}$ precision), and 1D isentropic relations ($T/T_0, p/p_0, \rho/\rho_0, A/A^*$). |
| [`unit_processes.py`](unit_processes.py) | Exact implementations of Anderson's 4 Unit Processes with 2nd-order modified Euler average characteristic slopes. |
| [`geometry.py`](geometry.py) | Nozzle wall classes (`BezierWall`, `ParabolicWall`, `SplineWall`) with analytical slopes $\theta(x)$ and characteristic intersection root-finding. |
| [`mln_designer.py`](mln_designer.py) | Anderson Section 11.5 Minimum-Length Nozzle (MLN) synthesizer with wave-cancellation contour design. |
| [`moc_solver.py`](moc_solver.py) | Direct MoC flow solver for any given diffusing nozzle geometry with wall reflections and centerline reflections. |
| [`thrust_calculator.py`](thrust_calculator.py) | Momentum flux, pressure thrust, mass flow rate, and thrust coefficient $C_T$ exit integration (Fernandes et al. 2023 Eqs. 17-22). |
| [`visualization.py`](visualization.py) | Publication-ready visualizer for characteristic wave grids, Mach number contours, and pressure fields. |
| [`validator.py`](validator.py) | Automated validation suite implementing Anderson Ch 11, Fernandes et al. (2023), and SU2 Euler CFD comparison tests. |
| [`run_validation.py`](run_validation.py) | Benchmark runner executing all 4 validation tests and generating comparison plots. |
| [`optimizer.py`](optimizer.py) | Differential Evolution optimization engine for thrust-maximizing bell nozzle contours with convexity constraints. |
| [`run_optimization.py`](run_optimization.py) | Optimization runner script comparing baseline vs optimal contours and generating diagnostic plots. |
| [`main.py`](main.py) | Main demonstration runner executing the textbook validation benchmark and arbitrary nozzle thrust evaluation. |

---

## Benchmark Highlights

- **Anderson Example 11.1 ($M_D = 2.40$)**: Exit area ratio error converges from $1.65\%$ ($N=7$) down to **$0.055\%$** ($N=100$).
- **SU2 Euler CFD Validation ($M_e = 3.00$)**: Matches CFD thrust coefficients $C_T$ across ambient pressures with **$<0.55\%$** discrepancy.
- **Mass Conservation**: Choked mass flow error is **$0.137\%$** (Divergence efficiency: $99.85\%$).
- **Differential Evolution Optimization**: Achieved a **$+69.52\%$** thrust increase for a fixed nozzle length ($L = 5.0\text{ m}$) by reducing exit divergence angle to **$1.09^\circ$**.

---

## Execution Commands

```bash
# Main demonstration
python anderson_moc_solver/main.py

# Benchmark validation against Anderson & CFD
python anderson_moc_solver/run_validation.py

# Differential Evolution nozzle contour optimization
python anderson_moc_solver/run_optimization.py
```
