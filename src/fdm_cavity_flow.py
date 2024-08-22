import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.ndimage import uniform_filter


class NavierStokesSolver:
    """
    This class implements the finite difference method for solving
    the 2D incompressible Navier-Stokes equations in a stepped cavity flow.

    The solver uses a staggered grid approach for the velocity components (V_X, V_Y)
    and pressure (p), and it iteratively solves the pressure Poisson equation to enforce
    the incompressibility condition (divergence-free velocity field).

    Attributes:
    -----------
    nx, ny : int
        Number of grid points in the x and y directions, respectively.
    Re : float
        Reynolds number of the flow.
    x, y : ndarray
        1D arrays representing the grid points in the x and y directions.
    dx, dy : float
        Grid spacing in the x and y directions.
    V_X, V_Y : ndarray
        2D arrays representing the velocity components in the x and y directions, respectively.
    p : ndarray
        2D array representing the pressure field.
    dt : float
        Time step size for the simulation.
    nt : int
        Number of time steps to run the simulation.
    nit : int
        Number of iterations for solving the pressure Poisson equation.
    """

    def __init__(self, nx=41, ny=41, x_edge=2, y_edge=2, Re=1.0):
        """
        Initializes the solver with the given grid size, domain size, and Reynolds number.

        Parameters:
        -----------
        nx, ny : int, optional
            Number of grid points in the x and y directions, respectively. Default is 41.
        x_edge, y_edge : float, optional
            Physical size of the domain in the x and y directions. Default is 2.
        Re : float, optional
            Reynolds number of the flow. Default is 1.0.
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.x = np.linspace(0, x_edge, nx)
        self.y = np.linspace(0, y_edge, ny)
        self.dx = x_edge / (nx - 1)
        self.dy = y_edge / (ny - 1)
        self.dt = 0.00001  # Small time step for stability
        self.nt = 1000  # Number of time steps
        self.nit = 100  # Iterations for pressure Poisson solver

        # Initialize the velocity and pressure fields
        self.V_X = np.zeros((ny, nx))
        self.V_Y = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.RHS = np.zeros((ny, nx))

        self.XX, self.YY = np.meshgrid(self.x, self.y)

    def poisson_pressure_RHS(self):
        """
        Computes the right-hand side (RHS) of the pressure Poisson equation.

        The RHS is derived from the incompressibility condition and the momentum equations.

        Returns:
        --------
        b : ndarray
            The RHS of the pressure Poisson equation.
        """
        b = np.zeros_like(self.p)
        try:
            b[1:-1, 1:-1] = (1 * (1 / self.dt *
                                  ((self.V_X[1:-1, 2:] - self.V_X[1:-1, 0:-2]) / (2 * self.dx) +
                                   (self.V_Y[2:, 1:-1] - self.V_Y[0:-2, 1:-1]) / (2 * self.dy)) -
                                  ((self.V_X[1:-1, 2:] - self.V_X[1:-1, 0:-2]) / (2 * self.dx)) ** 2 -
                                  2 * ((self.V_X[2:, 1:-1] - self.V_X[0:-2, 1:-1]) / (2 * self.dy) *
                                       (self.V_Y[1:-1, 2:] - self.V_Y[1:-1, 0:-2]) / (2 * self.dx)) -
                                  ((self.V_Y[2:, 1:-1] - self.V_Y[0:-2, 1:-1]) / (2 * self.dy)) ** 2))
        except FloatingPointError as e:
            print(f"Floating point error in RHS computation: {e}")
        return b

    def poisson_pressure_steps_solver(self, b):
        """
        Solves the pressure Poisson equation iteratively using the Jacobi method.

        Parameters:
        -----------
        b : ndarray
            The RHS of the pressure Poisson equation.
        """
        pn = np.empty_like(self.p)
        pn = self.p.copy()

        np.seterr(all='raise')

        try:
            for _ in range(self.nit):
                pn = self.p.copy()
                self.p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * self.dy ** 2 +
                                       (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * self.dx ** 2) /
                                      (2 * (self.dx ** 2 + self.dy ** 2)) -
                                      self.dx ** 2 * self.dy ** 2 /
                                      (2 * (self.dx ** 2 + self.dy ** 2)) * b[1:-1, 1:-1])

                # Apply boundary conditions for pressure
                self.p[:, -1] = 0  # dp/dy = 0 at x = 2
                self.p[0, :] = self.p[1, :]  # dp/dy = 0 at y = 0
                self.p[:, 0] = self.p[:, 1]  # dp/dx = 0 at x = 0
                self.p[-1, :] = 0  # p = 0 at y = 2 (top boundary)

        except FloatingPointError as e:
            print(f"Floating point error in pressure solver: {e}")

    def cavity_flow(self):
        """
        Simulates the cavity flow by iteratively solving the Navier-Stokes equations
        and the pressure Poisson equation.
        """
        un = np.empty_like(self.V_X)
        vn = np.empty_like(self.V_Y)

        for n in range(self.nt):
            un = self.V_X.copy()
            vn = self.V_Y.copy()

            b = self.poisson_pressure_RHS()
            self.poisson_pressure_steps_solver(b)

            self.V_X[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                                    un[1:-1, 1:-1] * self.dt / self.dx *
                                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                                    vn[1:-1, 1:-1] * self.dt / self.dy *
                                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                                    self.dt / (2 * self.dx) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
                                    self.dt / self.Re * (
                                    self.dx ** -2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                    self.dy ** -2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            self.V_Y[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                                    un[1:-1, 1:-1] * self.dt / self.dx *
                                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                                    vn[1:-1, 1:-1] * self.dt / self.dy *
                                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                                    self.dt / (2 * self.dy) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
                                    self.dt / self.Re * (
                                    self.dx ** -2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                    self.dy ** -2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            # Apply boundary conditions for velocity
            self.V_X[0, :] = 0  # Bottom boundary (no-slip)
            self.V_X[22:, 0] = -0.5 * (self.y[22:] - self.y[22]) * (
            self.y[22:] - self.y[-1])  # Quadratic inflow on the left wall
            self.V_X[0:22, 0] = 0  # No flow on the left wall below the step
            self.V_X[:, -1] = self.V_X[:, -2]  # Right wall (no-slip)
            self.V_X[-1, :] = 0  # Top boundary (no-slip)
            self.V_X[0:24, 0:12] = 0  # No flow in the bottom left corner

            self.V_Y[0, :] = 0  # Bottom boundary (no-slip)
            self.V_Y[-1, :] = 0  # Top boundary (no-slip)
            self.V_Y[:, 0] = 0  # Left boundary (no-slip)
            self.V_Y[:, -1] = self.V_Y[:, -2]  # Right wall (no-slip)
            self.V_Y[0:24, 0:12] = 0  # No flow in the bottom left corner

    def process_pressure_field(self):
        """
        Processes the pressure field by trimming negative values, applying a moving average filter,
        and setting pressure to NaN where velocity is zero.
        """
        # Trim negative pressure values
        self.p = np.clip(self.p, 0, None)

        # Apply a moving average filter to smooth the pressure field
        self.p = uniform_filter(self.p, size=7)

        # Set pressure to NaN where both velocity components are zero
        self.p[(self.V_X == 0) & (self.V_Y == 0)] = np.nan


# Instantiate and run the solver
ne = NavierStokesSolver(nx=81, ny=41, x_edge=2, y_edge=2, Re=0.5)
ne.cavity_flow()

# Process the pressure field
ne.process_pressure_field()

# Extract the results
V_X = ne.V_X
V_Y = ne.V_Y
Pressure = ne.p
X = ne.XX
Y = ne.YY

# Plot the results
fig = plt.figure(figsize=(11, 7), dpi=100)

# Plotting the pressure field as a contour plot
plt.contourf(X, Y, Pressure, alpha=0.5, cmap='cubehelix')
plt.colorbar()

# Plotting the pressure field outlines
plt.contour(X, Y, Pressure, cmap='cubehelix')

# Plotting the velocity field using quiver plot
plt.quiver(X[::2, ::2], Y[::2, ::2], V_X[::2, ::2], V_Y[::2, ::2])

# Labeling the axes
plt.xlabel('X')
plt.ylabel('Y')

# Display the plot
plt.title('Cavity Flow Velocity and Pressure Field')
plt.show()
