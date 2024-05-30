import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import gc
import matplotlib.colors as mcolors
from Body import Body
import numpy as np

def initialize_bodies_randomly(num_bodies, seed):

  """
  Initializes a list of Body objects with random masses, positions, and velocities.

  Args:
      num_bodies (int): The number of bodies to create.
      seed (int): The random seed to ensure reproducibility of random values.

  Returns:
      list: A list of Body objects representing the simulated bodies.

  This function generates a list of `Body` objects with random properties suitable
  for an n-body simulation. It sets ranges for mass, position, and velocity and
  uses NumPy's random number generation capabilities.
  """

  mass_range = (1.0e24, 6.0e24)
  position_range = (-1e11, 1e11)
  velocity_range = (-50, 200)

  np.random.seed(seed)

  masses = np.random.uniform(low=mass_range[0], high=mass_range[1], size=num_bodies)
  positions = np.random.uniform(low=position_range[0], high=position_range[1], size=(num_bodies, 2))
  velocities = np.random.uniform(low=velocity_range[0], high=velocity_range[1], size=(num_bodies, 2))

  bodies = []
  for i in range(num_bodies):

    body = Body(masses[i], positions[i], velocities[i])
    bodies.append(body)

  return bodies

def makegif(positions, colors, fps=15, interval=500, downsample_factor=10):

  """
  Creates an animated GIF visualizing the motion of bodies in an n-body simulation.

  Args:
      positions (list): A list of NumPy arrays, where each inner array represents the positions of all bodies at a specific time step.
      colors (list): A list of colors to assign to each body in the animation (length should match the number of bodies).
      fps (int, optional): The frames per second for the animation (default: 15).
      interval (int, optional): The delay (in milliseconds) between frames in the animation (default: 500).
      downsample_factor (int, optional): Factor to downsample the positions list for efficiency (default: 10).

  This function takes a list of body positions and colors and creates a 3D animated GIF visualization
  of their motion over time.
  """

  positions = np.array(positions, dtype=np.float32)  # Use float32 to reduce memory usage

  # Downsample positions
  positions = positions[:, ::downsample_factor, :]

  # Number of bodies and time steps
  num_bodies = positions.shape[0]
  num_steps = positions.shape[1]

  # Normalize positions
  x_min, x_max = np.min(positions[:, :, 0]), np.max(positions[:, :, 0])
  y_min, y_max = np.min(positions[:, :, 1]), np.max(positions[:, :, 1])

  # Normalize to the range (-1, 1)
  positions[:, :, 0] = 2 * (positions[:, :, 0] - x_min) / (x_max - x_min) - 1
  positions[:, :, 1] = 2 * (positions[:, :, 1] - y_min) / (y_max - y_min) - 1

  # Create z-coordinates (e.g., setting all z-coordinates to 0)
  z_positions = np.zeros((num_bodies, num_steps), dtype=np.float32)

  # Set up the figure, axis, and plot element for the animation
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim((-1, 1))
  ax.set_ylim((-1, 1))
  ax.set_zlim((-1, 1))

  # Initialize scatter plot with the first set of positions
  scatter = ax.scatter(positions[:, 0, 0], positions[:, 0, 1], z_positions[:, 0], c = colors)

  # Generator function to yield frames
  def frame_generator(frame):

    scatter._offsets3d = (positions[:, frame, 0], positions[:, frame, 1], z_positions[:, frame])
    gc.collect()  # Force garbage collection to free memory
    return scatter,

  # Create the animation
  ani = animation.FuncAnimation(fig, frame_generator, frames=num_steps, interval=interval, blit=True)

  # Save the animation as a GIF using Pillow writer
  ani.save('n_body_animation_3d.gif', writer='pillow', fps=fps)

  plt.show()
  
def compute_acceleration(bodies):

  """
  Calculates the acceleration of each body in a list of bodies due to the gravitational interaction
  with all other bodies in the list.

  Args:
    bodies (list): A list of Body objects representing physical objects in the simulation.

  This function iterates through each body in the list and calculates the gravitational force exerted on
  it by all other bodies. The force is then used to compute the acceleration using Newton's second law
  (force = mass * acceleration).

  **Constants:**
    G (float): Gravitational constant (6.67430e-11 m^3 kg^-1 s^-2)
    epsilon (float): Small value added to the squared distance to avoid division by zero, smoothing (1e-4)
  """

  G = 6.67430e-11  # Gravitational constant
  epsilon = 1e-4     # Small value to avoid division by zero

  num_bodies = len(bodies)  # Get the number of bodies in the list

  # Loop through each body in the list
  for i in range(num_bodies):
    # Initialize acceleration as a zero vector with dimensionality 2 (assuming 2D space)
    bodies[i].acceleration = np.zeros(2)

    # Loop through all other bodies (excluding the current body)
    for j in range(num_bodies):

      if j != i:  # Skip the current body to avoid self-interaction

        # Calculate the difference vector between the positions of body i and body j
        diff = bodies[j].position - bodies[i].position

        # Calculate the squared distance between the bodies, adding a small value to avoid division by zero
        dist_sq = np.dot(diff, diff) + epsilon**2

        # Calculate the magnitude of the gravitational force using the gravitational constant, masses, and squared distance
        force_mag = G * bodies[i].mass * bodies[j].mass / dist_sq

        # Calculate the direction of the gravitational force by normalizing the difference vector
        force_dir = diff / np.sqrt(dist_sq)

        # Update the acceleration of body i by adding the contribution from body j
        # (force divided by mass to get acceleration according to Newton's second law)
        bodies[i].acceleration += force_mag * force_dir / bodies[i].mass


def generate_n_colors(n):
    """
    Generate a list of n distinct colors.

    Parameters:
    n (int): Number of colors to generate

    Returns:
    List of n colors in hexadecimal format
    """
    # Use the 'hsv' colormap to generate n colors
    colors = plt.cm.hsv(np.linspace(0, 1, n))

    # Convert the colors from RGBA to hexadecimal format
    hex_colors = [mcolors.to_hex(color) for color in colors]

    return hex_colors
    
def load_test_case(test_case, N = 4):

  G = 6.67430e-11

  #################################################################################################################
  if test_case == "solar system" or test_case == "solar_system":

    sun = Body(1.989e30, np.array([0.0,0.0], dtype=np.float64), np.array([0.0,0.0], dtype=np.float64))
    mercury = Body(3.301e23, np.array([0.387 * 1.496e11, 0.0], dtype=np.float64), np.array([0.0, 47362.], dtype=np.float64))
    venus = Body(4.867e24, np.array([0.723 * 1.496e11, 0.0], dtype = np.float64), np.array([0.0, 35024], dtype = np.float64))
    earth = Body(5.972e24, np.array([1.496e11, 0], dtype = np.float64), np.array([0, 29780], dtype = np.float64))
    mars = Body(6.417e23, np.array([1.523 * 1.496e11, 0], dtype=np.float64), np.array([0, 24007], dtype=np.float64))
    jupiter = Body(1.898e27, np.array([5.203 * 1.496e11, 0], dtype=np.float64), np.array([0, 13070], dtype=np.float64))
    saturn = Body(5.683e26, np.array([9.537 * 1.496e11, 0], dtype = np.float64), np.array([0, 9690], dtype = np.float64))
    uranus = Body(8.681e25, np.array([19.19 * 1.496e11, 0], dtype = np.float64), np.array([0, 6800], dtype = np.float64))
    neptune = Body(1.024e26, np.array([30.07 * 1.496e11, 0], dtype = np.float64), np.array([0, 5430], dtype = np.float64))

    solar_system = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

    return solar_system[:N+1]

  ###############################################################################################################
  elif test_case == "earth moon" or test_case == "earth_moon":

    mass_earth = 5.972e24  # Mass of Earth in kg
    position_earth = np.array([0, 0], dtype=np.float64)  # Initial position of Earth at the origin
    velocity_earth = np.array([0, 0], dtype=np.float64)  # Initial velocity of Earth at rest

    mass_moon = 7.342e22  # Mass of the Moon in kg
    position_moon = np.array([384400e3, 0], dtype=np.float64) # Initial position of the Moon (approximately 384,400 km from Earth along the x-axis)
    velocity_moon = np.array([0, np.sqrt(G * mass_earth / np.linalg.norm(position_moon))], dtype=np.float64) # Initial velocity of the Moon (circular orbital velocity around Earth)

    earth_m = Body(mass_earth, position_earth, velocity_earth)
    moon = Body(mass_moon, position_moon, velocity_moon)

    earth_moon = [earth_m, moon]
    return earth_moon

  ###############################################################################################################
  elif test_case == "pluto charon" or test_case == "pluto_charon":
    mass_pluto = 1.309e22  # Mass of Pluto in kg
    position_pluto = np.array([0, 0], dtype=np.float64)  # Initial position of Pluto at the origin
    velocity_pluto = np.array([0, 0], dtype=np.float64)  # Initial velocity of Pluto at rest

    mass_charon = 1.586e21  # Mass of Charon in kg
    position_charon = np.array([19570e3, 0], dtype=np.float64) # Initial position of Charon relative to Pluto (approximately 19,570 km from Pluto along the x-axis)
    velocity_charon = np.array([0, np.sqrt(G * mass_pluto / np.linalg.norm(position_charon))], dtype=np.float64) # Initial velocity of Charon (circular orbital velocity around Pluto)

    pluto = Body(mass_pluto, position_pluto, velocity_pluto)
    charon = Body(mass_charon, position_charon, velocity_charon)

    pluto_charon = [pluto, charon]

    return pluto_charon