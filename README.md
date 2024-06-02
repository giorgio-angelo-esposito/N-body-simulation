# N-body Simulation

The code for an N-body simulation in Python. This is the project for the Computetional Models for Complex System exam.

# Project structure

Both Jupyter notebook and Python script file (.py) are provided to facilitate different preferences.
The script files are:

* `Body.py`: this is the class definition of a Body.
* `simulation.py`: this script contains the main simulation algorithm. 
* `simulation_methods.py`: this script contains the implementation of three methods for solving ODEs: Euler, Runge-Kutta and Adams-Bashford.
* `simulation_utils.py`: this script contains different utility functions used in the simulation, like the code for computing the gravitational force between the bodies, create a .gif file that shows the motion of each body, and so on.
* `main.py`: this is the file you call for running the simulation.

# Required libraries

* `numpy`
* `matplotlib`
* `mpl_toolkits`
* `gc`

# Installation
```bash
git clone https://github.com/giorgio-angelo-esposito/N-body-simulation.git cd N-body-simulation
```

# Usage

To run the simulation, navigate to the root directory of the repository and execute the `main.py` file. You can use the `--help` flag to see the parameters that can be set:

```bash
python main.py --help
```

If not setted, the default parameters will be used.

# Some results
- **Solar System (4 planets)
![Solar System](https://github.com/giorgio-angelo-esposito/N-body-simulation/blob/main/gifs/solar_system_4.gif)

- **Sun-Earth-Moon
![Sun-Earth-Moon](https://github.com/giorgio-angelo-esposito/N-body-simulation/blob/main/gifs/sun_earth_moon.gif)

- ** Earth-Moon
![Earth-Moon](https://github.com/giorgio-angelo-esposito/N-body-simulation/blob/main/gifs/earth_moon.gif)

- ** Pluto-Charon
![Pluto-Charon](https://github.com/giorgio-angelo-esposito/N-body-simulation/blob/main/gifs/pluto_charon.gif)
