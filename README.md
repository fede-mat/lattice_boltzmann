<p align="center">
  <img src="animazione_2.gif" alt="Animation of the Lattice Boltzmann simulation of a dissolving body" width="900">
</p>

# Lattice Boltzmann Simulation of a Dissolving Body

This project implements a numerical model of a solid body dissolving in a fluid flow. The simulation couples an incompressible flow solver with an advection–diffusion model for the dissolved substance and updates the obstacle geometry according to the local diffusive flux.

The implementation is written in **Python** using **NumPy**. It is based on the *Lattice Boltzmann methods* course materials at the **Technical University of Munich (TUM), Summer Semester 2026**, in particular the sugar-ball dissolution exercise (labelled *Exercise Sheet P3a* in the sheet and *Programming Tutorial 3b* in the Python template). The code builds on that course assignment and should be read as a course project, rather than as a general-purpose CFD solver.

> **Animation:** `animazione_2.gif` is displayed at the top of this page. Place the GIF in the same directory as this `README.md` so that GitHub can render it.

## Contents

- [Model equations](#model-equations)
- [Numerical method](#numerical-method)
- [Boundary conditions](#boundary-conditions)
- [How the obstacle dissolves](#how-the-obstacle-dissolves)
- [Implementation and workflow](#implementation-and-workflow)
- [Parameters](#parameters)
- [Requirements and execution](#requirements-and-execution)
- [Output](#output)
- [Limitations](#limitations)
- [References](#references)

## Model equations

Let $\Omega_f(t)\subset\mathbb{R}^2$ denote the fluid region at time $t$, and let $\Gamma(t)$ be the surface of the dissolving obstacle. The unknowns are the fluid velocity $\mathbf{u}(\mathbf{x},t)$, pressure $p(\mathbf{x},t)$, and dissolved-substance concentration $C(\mathbf{x},t)$. The obstacle geometry—and therefore the fluid domain—changes over time.

### Incompressible flow

The fluid is modelled by the incompressible Navier–Stokes equations:

$$
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
+ \frac{1}{\rho}\nabla p
- \nu\Delta\mathbf{u}=0
\qquad\text{in }\Omega_f(t),
$$

$$
\nabla\cdot\mathbf{u}=0
\qquad\text{in }\Omega_f(t).
$$

Here, $\rho$ is the fluid density and $\nu$ the kinematic viscosity.

### Advection–diffusion of the concentration

The concentration of dissolved material satisfies

$$
\frac{\partial C}{\partial t}
+ \mathbf{u}\cdot\nabla C
- D\Delta C=0
\qquad\text{in }\Omega_f(t),
$$

where $D$ is the diffusivity. The velocity field transports the concentration by advection, while diffusion spreads it through the fluid.

### Dissolution and moving boundary

The obstacle surface recedes according to the prescribed flux law

$$
v_n=-D\nabla C\cdot\mathbf{n}
\qquad\text{on }\Gamma(t),
$$

where $\mathbf{n}$ is the unit normal used by the implementation and $v_n$ is the signed normal recession rate. This relation connects the concentration gradient at the interface to the local change in geometry. In the numerical implementation, the interface is represented on the lattice and its geometry changes by removing solid cells once their accumulated recession measure crosses a threshold.

The coupling is therefore two-way at the system level:

1. The flow velocity $\mathbf{u}$ enters the advection term in the concentration equation.
2. The concentration gradient determines the local recession rate.
3. The updated obstacle changes the fluid region and its boundary conditions, affecting subsequent flow and transport.

## Numerical method

### D2Q9 Lattice Boltzmann model

The code uses a two-dimensional **D2Q9 lattice**, with nine discrete velocity directions $\mathbf{c}_i$ and associated weights $w_i$. Two sets of distribution functions are evolved:

- $f_i$: hydrodynamic populations used to recover the fluid density and velocity;
- $g_i$: scalar populations used to recover the concentration field.

The macroscopic fields are reconstructed from the populations as

$$
\rho=\sum_{i=0}^{8} f_i,
\qquad
\rho\mathbf{u}=\sum_{i=0}^{8} f_i\mathbf{c}_i,
\qquad
C=\sum_{i=0}^{8} g_i.
$$

The implementation uses a single-relaxation-time BGK collision operator. For the flow field,

$$
f_i^{\star}=f_i-\omega\left(f_i-f_i^{\mathrm{eq}}\right),
$$

with equilibrium distribution

$$
f_i^{\mathrm{eq}}
=w_i\rho\left[
1+3(\mathbf{c}_i\cdot\mathbf{u})
+\frac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2
-\frac{3}{2}\lVert\mathbf{u}\rVert^2
\right].
$$

For concentration, the analogous update is

$$
g_i^{\star}=g_i-\omega_C\left(g_i-g_i^{\mathrm{eq}}\right),
$$

where

$$
g_i^{\mathrm{eq}}
=w_i C\left[
1+3(\mathbf{c}_i\cdot\mathbf{u})
+\frac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2
-\frac{3}{2}\lVert\mathbf{u}\rVert^2
\right].
$$

In lattice units, the relaxation parameters are related to viscosity and diffusivity by

$$
\omega=\frac{1}{3\nu+\tfrac12},
\qquad
\omega_C=\frac{1}{3D+\tfrac12}.
$$

After collision, the **streaming step** shifts each population along its corresponding lattice velocity. In the Python code this is implemented with `numpy.roll`. The flow and concentration fields share the same velocity field, which supplies the advective coupling.

## Boundary conditions

The implementation follows the boundary-condition setup in the course exercise:

| Boundary | Flow | Concentration |
|---|---|---|
| Inlet | Prescribed Poiseuille velocity profile, implemented with a Zou–He-type reconstruction | $C_{\mathrm{in}}=0$, applied with anti-bounce-back |
| Outlet | Constant-density/pressure treatment | Missing outgoing-side populations are set using the zero-value treatment in the code |
| Top and bottom walls | Bounce-back, enforcing no-slip | Bounce-back |
| Obstacle surface | Half-way bounce-back, enforcing no-slip | Anti-bounce-back with $C_{\mathrm{body}}=1$ |

The concentration boundary values represent a source of dissolved material at the obstacle surface and zero concentration at the inflow. The boundary populations are reconstructed direction by direction using the D2Q9 stencil.

## How the obstacle dissolves

The obstacle is stored as a Boolean array: `True` denotes a solid cell and `False` a fluid cell. The geometry-dependent boundary masks are calculated by `obstacle_bb_directions`, which identifies lattice links adjacent to the obstacle as well as neighbouring inner cells.

At each time step, the code:

1. Constructs an approximate surface normal from the obstacle boundary masks.
2. Estimates the concentration-gradient-related flux from the first moment of the scalar populations.
3. Projects that flux onto the normal to obtain the signed local recession rate.
4. Transfers/averages the boundary recession value to neighbouring inner cells.
5. Accumulates the recession measure in `recession_progress`.
6. Converts solid cells to fluid cells once the threshold is exceeded, then recomputes the boundary masks for the updated geometry.

Because the geometry is represented on a discrete lattice, the surface cannot move by an arbitrary fraction of a cell. The code therefore accumulates the recession measure and updates the geometry in whole-cell increments. This is a simple approximation to a continuously moving interface.

## Implementation and workflow

The main script is `LBM_SBD_modified.py`. Its structure is organised around the following operations:

- **Parameters and geometry:** set the lattice dimensions, flow parameters and initial circular obstacle.
- **D2Q9 stencil:** define lattice velocities, weights and opposite directions.
- **Macroscopic fields:** recover density, velocity and concentration from the distributions.
- **Collision and streaming:** advance the two population fields with BGK relaxation and lattice streaming.
- **Boundary conditions:** impose the inlet, outlet, channel-wall and obstacle conditions.
- **Interface update:** estimate the normal and concentration flux, accumulate recession, and update the obstacle mask.
- **Visualisation and output:** show or save snapshots and export the final fields in VTK format.

## Parameters

The uploaded Python script uses the following default setup:

| Parameter | Default | Meaning |
|---|---:|---|
| `Nx` | 400 | Number of lattice cells in the streamwise direction |
| `Ny` | 100 | Number of lattice cells in the transverse direction |
| `Nt` | 10000 | Number of time steps |
| `uMax` | 0.1 | Maximum inlet velocity in lattice units |
| `Re` | 100 | Reynolds number |
| `Pr` | 0.5 | Prandtl number used to set the diffusivity-to-viscosity ratio |
| `plot_interval` | 25 | Snapshot/visualisation interval, in time steps |
| `recession_threshold` | 10 | Threshold used for the lattice-cell geometry update |

The code derives viscosity and diffusivity from

$$
\mathrm{Re}=\frac{u_{\max}(2R)}{\nu},
\qquad
\mathrm{Pr}=\frac{\nu}{D},
$$

so that

$$
\nu=\frac{u_{\max}(2R)}{\mathrm{Re}},
\qquad
D=\frac{\nu}{\mathrm{Pr}},
$$

where $R$ is the initial obstacle radius in lattice units. These are lattice-unit parameters; they are not automatically SI values.

## Requirements and execution

### Requirements

- Python
- NumPy
- Matplotlib
- `pyevtk` for VTK output

Install the Python packages with:

```bash
python -m pip install numpy matplotlib pyevtk
```

### Run the simulation

From the directory containing the script, execute:

```bash
python LBM_SBD_modified.py
```

The current script uses hard-coded relative output paths beginning with `coding_session/second_problem/`. Run it from a directory where those paths are appropriate, or edit the output paths in the script to match your repository layout.

The Boolean `measure_performance` controls the output mode:

- `True`: saves PNG snapshots at the configured interval and reports elapsed time and MLUPS;
- `False`: opens the interactive Matplotlib visualisation during the simulation.

## Output

Depending on the selected mode, the script produces:

- **PNG snapshots** of the velocity magnitude and concentration fields, with the obstacle masked out;
- **VTK output** for the final density, velocity components and concentration, which can be inspected with tools such as ParaView;
- **Performance information**, including total elapsed time and lattice updates per second (MLUPS).

The GIF shown at the top of this README is a presentation asset and is not generated automatically by the script. To have it appear on GitHub, add `animazione_2.gif` to the repository root alongside `README.md`.

## Limitations

- The implementation is a two-dimensional, lattice-unit simulation and uses a relatively simple BGK Lattice Boltzmann scheme.
- The solid surface is represented by a Boolean lattice mask; geometry changes occur in whole-cell increments rather than through a continuously tracked interface.
- Normals and concentration fluxes at the interface are approximated from lattice populations and neighbouring-cell masks.
- The model and boundary conditions follow the course exercise and are not intended to cover every physical effect involved in dissolution.
- Numerical results depend on grid resolution, relaxation parameters, boundary treatment and the chosen geometry-update threshold; these should be checked when changing the default setup.

## References

1. **TUM course material.** *Lecture on Lattice Boltzmann methods*, Summer Semester 2026; sugar-ball dissolution exercise (Exercise Sheet P3a / Programming Tutorial 3b). Instructors listed in the supplied material: Prof. Dr. Barbara Wohlmuth and Dr. Stephan Lunowa.
2. J. M. Huang, M. N. J. Moore, and L. Ristroph, “Shape dynamics and scaling laws for a body dissolving in fluid flow,” *Journal of Fluid Mechanics*, **765** (2015), R3. [https://doi.org/10.1017/jfm.2014.718](https://doi.org/10.1017/jfm.2014.718)
