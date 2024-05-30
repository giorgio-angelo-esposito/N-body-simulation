from Body import Body
import numpy as np
from simulation_utils import compute_acceleration

def Euler_method(bodies, dt):

  """
  Implements the Euler method to update the positions and velocities of a list of bodies
  over a small time step (dt).

  Args:
    bodies (list): A list of Body objects representing physical objects in the simulation.
    dt (float): The time step for the simulation.

  This function uses the Euler method, a numerical integration technique, to update the
  positions and velocities of each body in the list based on their current accelerations
  and the given time step. The Euler method assumes a constant acceleration within the
  time step, which can lead to some inaccuracies for larger time steps.
  """

  num_bodies = len(bodies)  # Get the number of bodies in the list

  # Loop through each body in the list
  for i in range(num_bodies):

    # Store the current velocity for later use (avoids overwriting)
    current_velocity = bodies[i].velocity

    # Update the velocity using the current acceleration and time step
    bodies[i].velocity += bodies[i].acceleration * dt

    # Update the position using the current velocity and time step
    bodies[i].position += current_velocity * dt
    
def Runge_Kutta_method(bodies, dt):

  """
  Implements the Runge-Kutta method (likely Runge-Kutta 4) to update the positions and velocities
  of a list of bodies over a small time step (dt).

  Args:
    bodies (list): A list of Body objects representing physical objects in the simulation.
    dt (float): The time step for the simulation.

  This function uses the Runge-Kutta method, a numerical integration technique known for its
  higher accuracy compared to the Euler method. It works by evaluating the derivatives (acceleration
  in this case) at multiple points within the time step and using a weighted average to update
  positions and velocities. This provides a more accurate representation of the motion within the
  time step.
  """

  num_bodies = len(bodies)  # Get the number of bodies in the list

  # Store initial positions and velocities to avoid overwriting during calculations
  initial_positions = [body.position.copy() for body in bodies]
  initial_velocities = [body.velocity.copy() for body in bodies]

  # **Stage 1 (k1 calculation):**
  #   - Calculate accelerations using the current state
  #   - Calculate temporary changes in velocity based on current acceleration and dt
  #   - Calculate temporary changes in position based on current velocity and dt
  compute_acceleration(bodies)
  k1_v = [body.acceleration * dt for body in bodies]
  k1_p = [body.velocity * dt for body in bodies]

  # **Stage 2 (k2 calculation):**
  #   - Update positions by half the temporary change from stage 1
  #   - Update velocities by half the temporary change from stage 1
  #   - Recalculate accelerations based on the updated state
  #   - Calculate temporary changes in velocity and position based on new accelerations and dt
  for i in range(num_bodies):
    bodies[i].position = initial_positions[i] + 0.5 * k1_p[i]
    bodies[i].velocity = initial_velocities[i] + 0.5 * k1_v[i]
  compute_acceleration(bodies)

  k2_v = [body.acceleration * dt for body in bodies]
  k2_p = [body.velocity * dt for body in bodies]

  # **Stage 3 (k3 calculation): Similar to Stage 2**
  #   - Update positions by half the temporary change from stage 2
  #   - Update velocities by half the temporary change from stage 2
  #   - Recalculate accelerations based on the updated state
  #   - Calculate temporary changes in velocity and position based on new accelerations and dt
  for i in range(num_bodies):
    bodies[i].position = initial_positions[i] + 0.5 * k2_p[i]
    bodies[i].velocity = initial_velocities[i] + 0.5 * k2_v[i]
  compute_acceleration(bodies)

  k3_v = [body.acceleration * dt for body in bodies]
  k3_p = [body.velocity * dt for body in bodies]

  # **Stage 4 (k4 calculation): Similar to Stage 1 but with positions updated from Stage 3**
  #   - Update positions by the full temporary change from stage 3
  #   - Update velocities by the full temporary change from stage 3
  #   - Recalculate accelerations based on the final updated state
  #   - Calculate temporary changes in velocity and position based on new accelerations and dt
  for i in range(num_bodies):

    bodies[i].position = initial_positions[i] + k3_p[i]
    bodies[i].velocity = initial_velocities[i] + k3_v[i]
  compute_acceleration(bodies)

  k4_v = [body.acceleration * dt for body in bodies]
  k4_p = [body.velocity * dt for body in bodies]

  # Update positions and velocities:
  #   - Use a weighted average of temporary changes from all stages for final updates
  for i in range(num_bodies):
    bodies[i].position = initial_positions[i] + (1/6) * (k1_p[i] + 2*k2_p[i] + 2*k3_p[i] + k4_p[i])
    bodies[i].velocity = initial_velocities[i] + (1/6) * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])

def initialize_histories(bodies, dt, order):

  """
  Initializes histories for the Adams-Bashforth method using Runge-Kutta steps.

  Args:
      bodies (list): A list of Body objects representing physical objects in the simulation.
      dt (float): The time step for the simulation.
      order (int): The order of the Adams-Bashforth method to be used (number of past time steps considered).

  This function prepares initial histories of accelerations and velocities for all bodies
  in the simulation, which are needed for the Adams-Bashforth method.

  **Process:**

  1. Create empty lists to store acceleration and velocity histories for each body.
  2. Perform a loop for 'order' number of iterations:
      - Use the Runge-Kutta method (assumed to be implemented elsewhere) to update the
        positions and velocities of the bodies based on their current accelerations and
        the time step.
      - After each Runge-Kutta step, for each body:
          - Append a copy of its current acceleration to its corresponding history list.
          - Append a copy of its current velocity to its corresponding history list.
  3. Return the initialized lists of acceleration and velocity histories for all bodies.
  """

  acceleration_histories = [[] for i in range(len(bodies))]  # List to store acceleration histories
  velocity_histories = [[] for i in range(len(bodies))]     # List to store velocity histories

  for _ in range(order):
    Runge_Kutta_method(bodies, dt)  # Update positions and velocities (assumed implemented elsewhere)

    for i in range(len(bodies)):

      acceleration_histories[i].append(bodies[i].acceleration.copy())
      velocity_histories[i].append(bodies[i].velocity.copy())

  return acceleration_histories, velocity_histories
  
def Adam_Bashford_method(bodies, dt, order, acceleration_histories, velocity_histories):

  """
  Performs an Adams-Bashforth update for an n-body simulation.

  Args:
      bodies (list of Body): List of Body objects representing physical objects in the simulation.
      dt (float): Time step for the simulation.
      order (int): Order of the Adams-Bashforth method (determines how many past values of accelerations and velocities to use).
      acceleration_histories (list): List storing past accelerations for each body (assumed corrected from the previous comment).
      velocity_histories (list): List storing past velocities for each body.

  Returns:
      list, list: Updated acceleration and velocity histories.

  This function implements the Adams-Bashforth method, a predictor-corrector method,
  to update the positions and velocities of bodies in an n-body simulation. It uses past
  values of accelerations and velocities stored in the provided histories to predict
  new values.
  """

  # Adams-Bashforth coefficients for orders 1 to 4
  adams_bashforth_coefficients = {
        1: [1],
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24]
  }

  # Check if the given order is supported
  if order not in adams_bashforth_coefficients:
    raise ValueError(f"Order {order} is not supported. Supported orders are 1, 2, 3, and 4.")

  b = np.array(adams_bashforth_coefficients[order])

  if acceleration_histories is None and velocity_histories is None:
    acceleration_histories, velocity_histories = initialize_histories(bodies, dt, order)

  for i, body in enumerate(bodies):
    # Insert the current acceleration at the beginning of the history
      if len(acceleration_histories[i]) != order:
        acceleration_histories[i].insert(0, body.acceleration.copy())

      # Ensure histories have correct lengths
      acceleration_histories[i] = acceleration_histories[i][:order]
      velocity_histories[i] = velocity_histories[i][:order]

      # Calculate new velocity
      new_velocity = body.velocity + dt * sum(b[j] * acceleration_histories[i][j] for j in range(order-1,-1,-1))

      # Calculate new position
      new_position = body.position + dt * sum(b[j] * velocity_histories[i][j] for j in range(order-1,-1,-1))

      # Update body's position and velocity
      body.position = new_position
      body.velocity = new_velocity

      # Update velocity history
      velocity_histories[i].insert(0, new_velocity.copy())

      # Keep only the most recent 'order - 1' entries in the velocity history
      velocity_histories[i] = velocity_histories[i][:order]
      acceleration_histories[i] = acceleration_histories[i][:order-1]

  return acceleration_histories, velocity_histories