from Body import Body
from simulation_utils import generate_n_colors, load_test_case, makegif, initialize_bodies_randomly, compute_acceleration
from simulation_methods import Adam_Bashford_method, Runge_Kutta_method, Euler_method
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def simulation_(num_bodies = 5, seed = 42, dt = 86400, duration = 20, method = "RK", plot = "S", AB_order = 2, make_gif = False, test_case = None):

  """
  Performs an n-body simulation and visualizes the motion of bodies.

  Args:
    num_bodies (int, optional): The number of bodies in the simulation (default: 5).
    seed (int, optional): Random seed for reproducibility (default: 42).
    dt (float, optional): Time step for the simulation (default: 86400 seconds, equivalent to one day).
    duration (float, optional): Simulation duration in days (default: 20).
    method (str, optional): Integration method for calculating motion ("RK" for Runge-Kutta, "E" for Euler, "AB" for Adams-Bashforth, default: "RK").
    plot (str, optional): Visualization type ("S" for scatter plot, "L" for line plot, default: "S").
    AB_order (int, optional): Order of the Adams-Bashforth method (used only if method="AB", default: 2).
    make_gif (bool, optional): Whether to create an animated GIF of the simulation (default: False).
    test_case (str, optional): Name of a test case file to load initial conditions (default: None).

  This function performs an n-body simulation and visualizes the motion of the bodies. It allows for
  customizing various aspects of the simulation and visualization.
  """

  print("Simulation parameters:")
  print("num_bodies: ", num_bodies)
  print("seed: ", seed)
  print("dt: ", dt)
  print("duration: ", duration*1e7)
  print("method: ", method)
  print("plot: ", plot)
  print("AB_order: ", AB_order)
  print("makegif: ", make_gif)
  print("test_case: ", test_case)

  if test_case is None:
    b = initialize_bodies_randomly(num_bodies, seed)
  else:
    b = load_test_case(test_case, num_bodies)

  for i, body in enumerate(b):
    print(f"Body {i}: Mass = {body.mass}, Position = {body.position}, Velocity = {body.velocity}")

  # Adjust velocities to center around a common point (e.g., center of mass)
  total_mass = sum(body.mass for body in b)
  center_of_mass_velocity = sum(body.mass * body.velocity for body in b) / total_mass

  for body in b:
    body.velocity -= center_of_mass_velocity

  # Print adjusted velocities
  print("\nAdjusted velocities to center around a common point:")
  for i, body in enumerate(b):
    print(f"Body {i}: Mass = {body.mass}, Position = {body.position}, Velocity = {body.velocity}")


  # Create a list to store positions of each body over time
  positions_over_time = [[body.position.copy()] for body in b]

  # Initialize simulation time and potentially scale duration
  t = 0
  duration *= 1e7  # Assuming duration is already defined elsewhere

  # Set up the 3D plot
  fig = plt.figure(figsize=(9, 9))
  ax = fig.add_subplot(projection='3d')

  # Generate unique colors for each body
  colors = generate_n_colors(len(b))

  # Main simulation loop
  while t < duration:

    # Update accelerations for all bodies
    compute_acceleration(b)

    # Choose and apply numerical integration method based on 'method'
    if method == "Runge-Kutta" or method == "RK":
      Runge_Kutta_method(b, dt)
    elif method == "Euler" or method == "E":
      Euler_method(b, dt)
    elif method == "Adams-Bashford" or method == "AB":

      if t == 0:
        acc, vel = Adam_Bashford_method(b, dt, AB_order, None, None)
      else:
        acc, vel = Adam_Bashford_method(b, dt, AB_order, acc, vel)

    # Update and store positions of each body
    for i in range(len(b)):
      positions_over_time[i].append(b[i].position.copy())

    t += dt  # Increment simulation time

  # Prepare data for plotting (assuming 3D with mostly zero z-coordinate)
  for i in range(len(b)):
    positions = np.array(positions_over_time[i])
    z = np.zeros(positions.shape[0])

    # Choose plot type based on 'plot' variable
    if plot == "scatter" or plot == "S":
      ax.scatter(positions[:, 0], positions[:, 1], z, label=f"Body {i}", s=1, color=colors[i])
    elif plot == "line" or plot == "L":
      ax.plot(positions[:, 0], positions[:, 1], z, label=f"Body {i}", color=colors[i])

  # Add legend and display plot
  ax.legend()
  print("Simulation ended")
  plt.show()

  # Create a GIF if 'make_gif' is True
  if make_gif:
    makegif(positions_over_time, colors)