# Navier-Stokes Solver for Stepped Cavity Flow

This project implements a finite difference method to solve the 2D incompressible Navier-Stokes equations for a stepped cavity flow. The code simulates the fluid flow inside a cavity with a step and visualizes the resulting velocity and pressure fields.

<p align="center">
  <img src="https://github.com/nircko/cavity_flow_fdm/blob/d5614d712b284fdeb111120bccdc48d61dbdc45d/data/cavity_flow.png" alt="Cavity Flow Result" width="400"/>
</p>



### Navier-Stokes Equations

The Navier-Stokes equations describe the motion of fluid substances such as liquids and gases. These equations are a set of nonlinear partial differential equations that express the conservation of momentum and mass in a fluid. For an incompressible, viscous fluid, the Navier-Stokes equations in two dimensions can be written as:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

Where:
- \( u \) and \( v \) are the velocity components in the \( x \) and \( y \) directions, respectively.
- \( p \) is the pressure field.
- \( \rho \) is the fluid density.
- \( \nu \) is the kinematic viscosity of the fluid.
- 
### Finite Difference Method

The finite difference method (FDM) is a numerical technique used to approximate solutions to differential equations by discretizing the equations over a grid. In this project, the FDM is used to solve the Navier-Stokes equations by approximating derivatives with finite differences. The pressure field is obtained by solving a Poisson equation derived from the incompressibility condition, and the velocity field is updated using the computed pressure.

## Prerequisites

To run this project, you need the following software installed:

- Python 3.6 or higher
- NumPy
- Matplotlib
- SciPy

You can install the necessary Python packages using pip:

```bash
pip install numpy matplotlib scipy
