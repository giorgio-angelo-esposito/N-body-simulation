import numpy as np

"""
Definition of the Body class:

This class represents a physical object with mass, position, and velocity
in a simulated environment. It provides the foundation for modeling
the motion of objects under the influence of forces (like gravity).

**Attributes:**

* mass (float): The mass of the body, typically in kilograms (kg).
* position (NumPy array): The current position of the body in
  the environment, represented as a 2D.
* velocity (NumPy array): The current velocity of the body,
  representing its speed and direction, typically represented as a vector
  with the same dimensionality as position.
* acceleration (NumPy array, initially 0): The current acceleration
  of the body, representing the change in velocity per unit time, typically
  initialized to zero and updated based on force calculations.
"""

class Body:

  def __init__(self, mass, position, velocity):

    """
    Initializes a new Body object.

    Args:
        mass (float): The mass of the body.
        position (NumPy array): The initial position of the body.
        velocity (NumPy array): The initial velocity of the body.
    """

    self.mass = mass
    self.position = position
    self.velocity = velocity
    self.acceleration = np.zeros(2)  # Initialize acceleration as a zero vector