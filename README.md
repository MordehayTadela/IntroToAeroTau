## NACA0012 analysis with vortex panel method
In this notebook , we apply the vortex panel method to a NACA0012 airfoil in order to compute the flow field and estimate the lift. By discretizing the airfoil surface into panels with distributed vortices, we enforce the no-penetration boundary condition and compute the circulation needed to evaluate the aerodynamic lift.
mport os
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plti
# read of the geometry from a data file
naca_filepath = 'naca0012.dat'
with open (naca_filepath, 'r') as file_name:
    x, y = np.loadtxt(file_name, dtype=float, delimiter='\t', unpack=True)

# plot the geometry
width = 10
plt.figure(figsize=(width, width))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.axis('equal')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1);
## Discretization into panels
class Panel:
    """
    Contains information related to a panel.
    """
    def __init__(self, xa, ya, xb, yb):
        """
        Initializes the panel.
        
        Sets the end-points and calculates the center, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is on the lower or upper surface of the geometry.
        Initializes the source-sheet strength, tangential velocity,
        and pressure coefficient to zero.
        
        Parameters
        ----------
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # control-point (center-point)
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # length of the panel
        
        # orientation of the panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)
        
        # location of the panel
        if self.beta <= np.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient
def define_panels(x, y, N=200):
    """
    Discretizes the geometry into panels using the 'cosine' method.
    
    Parameters
    ----------
    x: 1D array of floats
        x-coordinate of the points defining the geometry.
    y: 1D array of floats
        y-coordinate of the points defining the geometry.
    N: integer, optional
        Number of panels;
        default: 40.
    
    Returns
    -------
    panels: 1D Numpy array of Panel objects
        The discretization of the geometry into panels.
    """
    R = (x.max() - x.min()) / 2  # radius of the circle
    x_center = (x.max() + x.min()) / 2  # x-coord of the center
    # define x-coord of the circle points
    x_circle = x_center + R * np.cos(np.linspace(0.0, 2 * np.pi, N + 1))
    
    x_ends = np.copy(x_circle)  # projection of the x-coord on the surface
    y_ends = np.empty_like(x_ends)  # initialization of the y-coord Numpy array

    x, y = np.append(x, x[0]), np.append(y, y[0])  # extend arrays using numpy.append
    
    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]
    
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])
    
    return panels
Now we can use this function, calling it with a desired number of panels whenever we execute the cell below. We also plot the resulting geometry.
N = 128                          # number of panels
panels = define_panels(x, y, N)  # discretizes of the geometry into panels

# plot the geometry and the panels
width = 10
plt.figure(figsize=(width, width))
plt.grid()
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.plot(x, y, color='k', linestyle='-', linewidth=2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
            np.append([panel.ya for panel in panels], panels[0].ya),
            linestyle='-', linewidth=1, marker='o', markersize=6, color='#CD2305')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.axis('equal')
## Freestream conditions
class Freestream:
    """
    Freestream conditions.
    """
    def __init__(self, u_inf=1.0, alpha=0):
        """
        Sets the freestream speed and angle (with the x-axis).
        
        Parameters
        ----------
        u_inf: float, optional
            Freestream speed;
            default: 1.0.
        alpha: float, optional
            Angle of attack in degrees;
            default: 0.0.
        """
        self.u_inf = u_inf
        self.alpha = np.radians(alpha)  # degrees --> radians
# define and creates the object freestream
u_inf = 1.0                            # freestream spee
alpha = 7                         # angle of attack (in degrees)
freestream = Freestream(u_inf, alpha)  # instantiation of the object freestream
## Flow tangency boundary condition
Enforcing the flow-tangency condition on each *control point* approximately makes the body geometry correspond to a dividing streamline (and the approximation improves if we represented the body with more and more panels). So, for each panel $i$, we make $u_n=0$ at $(x_{c_i},y_{c_i})$, which leads to the equation derived in the previous lesson:

$$
\begin{align*}
0 &= V_\infty \cos \left(\alpha-\beta_i\right)- \sum_{j=1,j\neq i}^N \frac{\gamma}{2\pi} \int_j \frac{\left(x_{c_i}-x_j\right)\frac{\partial y_{c_i}}{\partial n_i} - \left(y_{c_i}-y_j\right)\frac{\partial x_{c_i}}{\partial n_i}}{\left(x_{c_i}-x_j\right)^2 + \left(y_{c_i}-y_j\right)^2} {\rm d}s_j
\end{align*}
$$

where $\frac{\partial x_{c_i}}{\partial n_i} = \cos \beta_i$ and $\frac{\partial y_{c_i}}{\partial n_i} = \sin \beta_i$.

def integral(x, y, panel, dxdz, dydz):
    """
    Evaluates the contribution of a panel at one point.
    
    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Source panel which contribution is evaluated.
    dxdz: float
        Derivative of x in the z-direction.
    dydz: float
        Derivative of y in the z-direction.
    
    Returns
    -------
    Integral over the panel of the influence at the given target point.
    """
    def integrand(s):
        return (((x - (panel.xa - np.sin(panel.beta) * s)) * dydz -
                 (y - (panel.ya + np.cos(panel.beta) * s)) * dxdz) /
                ((x - (panel.xa - np.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + np.cos(panel.beta) * s))**2))
    return integrate.quad(integrand, 0.0, panel.length)[0]
## Building the linear system
Here, we build and solve the linear system of equations of the form

$$
\begin{equation}
[A][\gamma] = [b]
\end{equation}
$$

In building the matrix, below, we call the `integral()` function with the correct values for the last parameters: $\cos \beta_i$ and $\sin\beta_i$, corresponding to a derivative in the normal direction.

Finally, we use `linalg.solve()` from NumPy to solve the system and find the strength of each panel.

def build_matrix(panels):
    """
    Builds the source matrix for vortex panels with Kutta condition.

    Parameters
    ----------
    panels: 1D array of Panel object
        The source panels.

    Returns
    -------
    A: 2D Numpy array of floats
        The source matrix (NxN matrix; N is the number of panels).
    """
    N = len(panels)
    A = np.empty((N, N), dtype=float)
    np.fill_diagonal(A, 0)

    for i, p_i in enumerate(panels[:-1]):  # עד N-1 (האחרונה שמורה לקוטה)
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(p_i.xc, p_i.yc, p_j,
                                                 np.cos(p_i.beta),
                                                 np.sin(p_i.beta))
    A[-1, 0] = 1.0
    A[-1, -1] = 1.0
    A[-1, 1:-1] = 0.0  # ensure only γ1 + γN = 0 is applied
    return A

def build_rhs(panels, freestream):
    """
    Builds the RHS vector with Kutta condition applied in the last row.

    Parameters
    ----------
    panels : list of Panel
    freestream : Freestream

    Returns
    -------
    b : ndarray of shape (N,)
    """
    N = len(panels)
    b = np.empty(N, dtype=float)

    for i, panel in enumerate(panels[:-1]):  # עד N-1
        b[i] = freestream.u_inf * np.cos(freestream.alpha - panel.beta)

    b[-1] = 0.0  # תנאי קוטה

    return b
A = build_matrix(panels)           # compute the influence matrix for vortex panels
b = build_rhs(panels, freestream)  # compute the freestream RHS (includes Kutta if needed)
# solve the linear system
gamma = np.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.gamma = gamma[i]         # store the vortex strength on each panel

### Lift Coefficient from the Kutta–Joukowski Theorem

From the **Kutta–Joukowski theorem**, the lift coefficient for the vortex panel method is given by:

$$
C_L = \frac{2\Gamma}{U_\infty c}
$$

where:

- $\Gamma$ is the **total circulation** around the airfoil, computed as the sum of all vortex strengths times their respective panel lengths:

$$
\Gamma = \sum_{j=1}^N \gamma_j \Delta s_j
$$

- $c$ is the **chord length** of the airfoil  
- $U_\infty$ is the **freestream velocity**

> The circulation is considered **positive for clockwise rotation** around the airfoil, consistent with the panel definition.

---
# Calculate the total circulation
gamma_total = sum(panel.gamma * panel.length for panel in panels)

# Compute chord length (assuming trailing edge - leading edge in x)
chord = max(panel.xb for panel in panels) - min(panel.xa for panel in panels)

# Compute the lift coefficient using Kutta–Joukowski theorem
cl = 2 * gamma_total / (freestream.u_inf * chord)

print(f"Lift Coefficient (C_L): {cl:.4f}")
alpha_range = np.linspace(0, 12, 25)  # זוויות התקפה במעלות
cl_values = []

for alpha_deg in alpha_range:
    freestream = Freestream(u_inf=1.0, alpha=alpha_deg)  # נניח שהמרה לרדיאנים בפנים

    A = build_matrix(panels)
    b = build_rhs(panels, freestream)
    gamma = np.linalg.solve(A, b)

    for i, panel in enumerate(panels):
        panel.gamma = gamma[i]

    gamma_total = sum(panel.gamma * panel.length for panel in panels)
    chord = max(panel.xb for panel in panels) - min(panel.xa for panel in panels)
    cl = 2 * gamma_total / (freestream.u_inf * chord)
    cl_values.append(cl)

# חישוב ערך תאורטי לפי תורת הכנף הדקה
alpha_rad = np.radians(alpha_range)  # נדרש לרדיאנים עבור הקו התאורטי
cl_theory = 2 * np.pi * alpha_rad

# ציור גרף
plt.figure(figsize=(8, 5))
plt.plot(alpha_range, cl_values, marker='o', label='Vortex Panel Method')
plt.plot(alpha_range, cl_theory, linestyle='--', label='Thin Airfoil Theory', color='r')
plt.xlabel(r'$\alpha$ [deg]', fontsize=14)
plt.ylabel(r'$C_L$', fontsize=14)
plt.title('Lift Coefficient vs Angle of Attack', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

#  Rankine half-body flow analysis
## Rankin half-body flow
**The Rankine half-body flow** is formed by superimposing a uniform (freestream) flow with a point source placed at the origin. The uniform flow moves in a straight line, while the source emits fluid radially outward. When combined, the two flows create a streamlined shape resembling a rounded hill, with a stagnation point where the two effects cancel. This idealized model helps visualize how fluid flows around blunt bodies in potential flow theory.


### Analytical Insight

A **source flow** represents fluid radiating outward from a point. It can be described using a velocity field $\left(x,y\right)$ at the source point:

$$\psi_\text{source}\left(x,y\right) = \frac{\sigma}{2\pi} \arctan \left(\frac{y-y_\text{source}}{x-x_\text{source}}\right)$$

and the velocity components are:

$$u_\text{source}\left(x,y\right) = \frac{\sigma}{2\pi} \frac{x-x_\text{source}}{\left(x-x_\text{source}\right)^2 + \left(y-y_\text{source}\right)^2}$$

$$v_\text{source}\left(x,y\right) = \frac{\sigma}{2\pi} \frac{y-y_\text{source}}{\left(x-x_\text{source}\right)^2 + \left(y-y_\text{source}\right)^2}$$

The stream function for this flow is:

$$
 \psi_{freestream} = U_\infty y
$$
 

The streamlines of the combination of a freestream and a source flow are:

$$\psi = \psi_{freestream}+\psi_{source} = U_\infty y + \frac{\sigma}{2\pi} \arctan \left(\frac{y-y_\text{source}}{x-x_\text{source}}\right)$$

And since differentiation is linear, the velocity field induced by the new flow pattern is simply the sum of the freestream velocity field and the source velocity field:

$$u = u_\text{freestream} + u_\text{source}$$
$$v = v_\text{freestream} + v_\text{source}$$

The stagnation points in the flow are points where the velocity is zero. To find their location, we solve the following equations:

$$u = 0 , \quad v = 0$$

which leads to (for a source at $\left(0,0 \right)$):

$$x_\text{stagnation} = x_\text{source} - \frac{\sigma}{2\pi U_\infty} = - \frac{\sigma}{2\pi U_\infty} $$

$$y_\text{stagnation} = y_\text{source} = 0$$

at the stagnation point:

$$\psi_{0} =  \frac{\sigma}{2\pi}$$


Therefore, the streamline expression at the face of the hill:

$$\psi(x, y) = U_\infty y + \frac{\sigma}{2\pi} \arctan \left( \frac{y}{x} \right) = \psi_0 $$

the **max** value of the hill ocours when $( x \to \infty )$ :

$$U_\infty y_\text{max} = \frac{\sigma}{2\pi}$$
$$y_\text{max} = \frac{\sigma}{2U_\infty} $$

Given a freestream velocity of **20 m/s** and a maximum hill height of **40 meters**, we can compute the source strength \( \sigma \) using the expression:

$$
y_\text{max} = \frac{\sigma}{2U_\infty}
$$

Solving for \( \sigma \):

$$
\sigma = 2U_\infty y_\text{max}
$$

Substituting the known values:

$$
\sigma = 2 \cdot 20 \cdot 40 = 1600\ \text{m}^2/\text{s}
$$
the maximum wind velocity is located at the point (x = 0, y = 40).

**Horizontal component** (x-direction) comes only from the uniform flow:
$u = U_\infty = 20$

**Vertical component** (y-direction) comes only from the source:
$v = \frac{\sigma}{2\pi y} = \frac{1600}{2\pi \cdot 40} = \frac{20}{\pi} \approx 6.37$

Total velocity magnitude:
$|\vec{V}| = \sqrt{u^2 + v^2} = \sqrt{20^2 + \left( \frac{20}{\pi} \right)^2} \approx \sqrt{400 + 40.5} \approx \sqrt{440.5} \approx 21.0$









## numerical implementation
We now proceed with the **numerical implementation** of the Rankine flow using the previously defined freestream velocity and maximum hill height. The stagnation point is assumed to be located at the origin.

import numpy
import math
from matplotlib import pyplot
# display figures in the Notebook
%matplotlib inline
N = 240                               # Number of points in each direction
x_start, x_end = -120.0, 120.0            # x-direction boundaries
y_start, y_end = -60.0, 60.0            # y-direction boundaries
x = numpy.linspace(x_start, x_end, N)    # 1D-array for x
y = numpy.linspace(y_start, y_end, N)    # 1D-array for y
X, Y = numpy.meshgrid(x, y)              # generates a mesh grid
numpy.shape(X)
**free stream:** u = 20 m/s , $v=0$.
u_inf = 20.0        # freestream speed

# compute the freestream velocity field
u_freestream = u_inf * numpy.ones((N, N), dtype=float)
v_freestream = numpy.zeros((N, N), dtype=float)

# compute the stream-function
psi_freestream = u_inf * Y
**source flow:** the velocity components are:

$u_\text{source}\left(x,y\right) = \frac{\sigma}{2\pi} \frac{x-x_\text{source}}{\left(x-x_\text{source}\right)^2 + \left(y-y_\text{source}\right)^2}$

$v_\text{source}\left(x,y\right) = \frac{\sigma}{2\pi} \frac{y-y_\text{source}}{\left(x-x_\text{source}\right)^2 + \left(y-y_\text{source}\right)^2}$
def get_velocity(strength, xs, ys, X, Y):
    """
    Returns the velocity field generated by a source/sink.
    
    Parameters
    ----------
    strength: float
        Strength of the source/sink.
    xs: float
        x-coordinate of the source (or sink).
    ys: float
        y-coordinate of the source (or sink).
    X: 2D Numpy array of floats
        x-coordinate of the mesh points.
    Y: 2D Numpy array of floats
        y-coordinate of the mesh points.
    
    Returns
    -------
    u: 2D Numpy array of floats
        x-component of the velocity vector field.
    v: 2D Numpy array of floats
        y-component of the velocity vector field.
    """
    u = strength / (2 * numpy.pi) * (X - xs) / ((X - xs)**2 + (Y - ys)**2)
    v = strength / (2 * numpy.pi) * (Y - ys) / ((X - xs)**2 + (Y - ys)**2)
    
    return u, v
??get_velocity
def get_stream_function(strength, xs, ys, X, Y):
    """
    Returns the stream-function generated by a source/sink.
    
    Parameters
    ----------
    strength: float
        Strength of the source/sink.
    xs: float
        x-coordinate of the source (or sink).
    ys: float
        y-coordinate of the source (or sink).
    X: 2D Numpy array of floats
        x-coordinate of the mesh points.
    Y: 2D Numpy array of floats
        y-coordinate of the mesh points.
    
    Returns
    -------
    psi: 2D Numpy array of floats
        The stream-function.
    """
    psi = strength / (2 * numpy.pi) * numpy.arctan2((Y - ys), (X - xs))
    
    return psi
for a source at $\left(0,0 \right)$), $\sigma = 4 \cdot 20 \cdot 40 = 1600\ \text{m}^2/\text{s}$
strength_source = 1600.0            # strength of the source
x_source, y_source = 0.0, 0.0   # location of the source

# compute the velocity field
u_source, v_source = get_velocity(strength_source, x_source, y_source, X, Y)

# compute the stream-function
psi_source = get_stream_function(strength_source, x_source, y_source, X, Y)
# superposition of the source on the freestream
u = u_freestream + u_source
v = v_freestream + v_source
psi = psi_freestream + psi_source

# plot the streamlines
width = 10
height = (y_end - y_start) / (x_end - x_start) * width
pyplot.figure(figsize=(width, height))
pyplot.grid(True)
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
pyplot.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->')
pyplot.scatter(x_source, y_source, color='#CD2305', s=80, marker='o')

# calculate the stagnation point
x_stagnation = x_source - strength_source / (2 * numpy.pi * u_inf)
y_stagnation = y_source

# display the stagnation point
pyplot.scatter(x_stagnation, y_stagnation, color='g', s=80, marker='o')

# display the dividing streamline
pyplot.contour(X, Y, psi, 
               levels=[-strength_source / 2, strength_source / 2], 
               colors='#CD2305', linewidths=2, linestyles='solid');
pyplot.show()
We first calculated the maximum velocity analytically at the point (x = 0, y = 40):

$
u = 20 m/s ,  v = 6.37 m/s  ,
$
$|\vec{V}| = \sqrt{u^2 + v^2} \approx 21.0 m/s
$

We will now compute the velocity numerically using the velocity field data.
# Evaluate velocity at point (x=0, y=40)
x_target = 0
y_target = 40

# Find nearest grid point
i = numpy.abs(x - x_target).argmin()  # index in x-direction (columns)
j = numpy.abs(y - y_target).argmin()  # index in y-direction (rows)

# Extract velocity components
u_val = u[j, i]
v_val = v[j, i]
V_mag = numpy.sqrt(u_val**2 + v_val**2)

# Print results
print(f"Velocity at (x=0, y=40):")
print(f"u = {u_val:.4f}, v = {v_val:.4f}, |V| = {V_mag:.4f}")
# compute the pressure coefficient field
cp = 1.0 - (u**2 + v**2) / u_inf**2

# plot the pressure coefficient field
width = 10
height = (y_end - y_start) / (x_end - x_start) * width
pyplot.figure(figsize=(1.1 * width, height))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)
contf = pyplot.contourf(X, Y, cp,
                        levels=numpy.linspace(-2.0, 1.0, 100), extend='both')
cbar = pyplot.colorbar(contf)
cbar.set_label('$C_p$', fontsize=16)
cbar.set_ticks([-2.0, -1.0, 0.0, 1.0])
pyplot.scatter(x_stagnation, y_stagnation, color='g', s=80, marker='o')
pyplot.scatter(x_source, y_source, color='#CD2305', s=80, marker='o')
pyplot.contour(X, Y, psi, 
               levels=[-strength_source / 2, strength_source / 2], 
               colors='#CD2305', linewidths=2, linestyles='solid');
pyplot.show()
# compute the velocity magnitude field
V_mag = numpy.sqrt(u**2 + v**2)

# plot the velocity magnitude field (auto-scaled)
width = 10
height = (y_end - y_start) / (x_end - x_start) * width
pyplot.figure(figsize=(1.1 * width, height))
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.xlim(x_start, x_end)
pyplot.ylim(y_start, y_end)

# use full range of values for contour levels
contf = pyplot.contourf(X, Y, V_mag,
                        levels=numpy.linspace(numpy.min(V_mag), numpy.max(V_mag), 100),
                        extend='both')
cbar = pyplot.colorbar(contf)
cbar.set_label('Velocity Magnitude $|\mathbf{V}|$ [m/s]', fontsize=16)

pyplot.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# נתונים
U_inf = 20                      # מהירות הזרימה
sigma = 1600                   # עוצמת המקור
psi_0 = sigma / (2 * np.pi)    # קו הזרם על שפת הגבעה
x_stagnation = -sigma / (2 * np.pi * U_inf)

# פונקציית שיור לפתרון ψ(x, y) = ψ0
def stream_function_residual(y, x, U_inf, sigma, psi_0):
    return U_inf * y + (sigma / (2 * np.pi)) * np.arctan2(y, x) - psi_0

# פתרון נומרי עבור y(x) על שפת הגבעה
def compute_accurate_y_on_hill(x_vals, U_inf, sigma, psi_0):
    y_vals = []
    for x in x_vals:
        y_guess = 1.0
        y_solution, = fsolve(stream_function_residual, y_guess, args=(x, U_inf, sigma, psi_0))
        y_vals.append(y_solution)
    return np.array(y_vals)

# חישוב המהירות הכוללת לאורך שפת הגבעה
def compute_velocity_on_accurate_hill(x_vals):
    y_vals = compute_accurate_y_on_hill(x_vals, U_inf, sigma, psi_0)
    u = U_inf + sigma / (2 * np.pi) * x_vals / (x_vals**2 + y_vals**2)
    v = sigma / (2 * np.pi) * y_vals / (x_vals**2 + y_vals**2)
    V = np.sqrt(u**2 + v**2)
    return V, y_vals

# טווח x (נמנעים מ-x קרוב מדי לאפס בגלל סינגולריות)
x_vals = np.linspace(x_stagnation + 0.1, 40, 500)
V_vals, y_vals = compute_velocity_on_accurate_hill(x_vals)

# גרף עם הגבלת ציר y עד 100
plt.plot(x_vals, V_vals)
plt.axvline(x=x_stagnation, color='gray', linestyle='--', label='Stagnation Point')
plt.ylim(0, 100)  # הגבלת ציר y
plt.xlabel("x [m]")
plt.ylabel("Velocity Magnitude [m/s]")
plt.title("Accurate Velocity Along Hill Surface (ψ = ψ₀)\n(Capped at 100 m/s)")
plt.grid(True)
plt.legend()
plt.show()

