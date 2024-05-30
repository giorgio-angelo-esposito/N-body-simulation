import argparse
from simulation import simulation_

# Definition of arguments
parser = argparse.ArgumentParser(description="N-nody simulation")
parser.add_argument("--num_bodies", type=int, default=5, help="Number of bodies in the simulation (default: 5)")
parser.add_argument("--seed", type=int, default=42, help="Seed for random generation (default: 42)")
parser.add_argument("--dt", type=float, default=86400, help="Temporal step (default: 86400, a day)")
parser.add_argument("--duration", type=float, default=20, help="Total duration of the simulation (default: 20)")
parser.add_argument("--method", choices=["RK", "Runge_Kutta", "AB", "Adams-Bashford", "E", "Euler"], default="RK", help="Method to solve the equations (default: RK)")
parser.add_argument("--AB_order", type=int, default=2, help="Order of the Adams-Bashford method (applied only if method = AB)")
parser.add_argument("--plot", choices=["S", "L"], default="S", help="Plot type: S (Scatter plot) o L (Normal plot) (default: S)")
parser.add_argument("--make_gif", action="store_true", help="Flag for the creation of an animation of the simulation (default: False)")
parser.add_argument("--test_case", choices=["solar_system", "earth_moon", "pluto_charon"], default=None, help="String for the load of a test_case. Available\nsolar_system, N\nearth_moon\npluto_charon\n (default: None)")

# Acquisition and validation of arguments
args = parser.parse_args()

# Execution of the simulation with the given arguments
simulation_(num_bodies=args.num_bodies, seed=args.seed, dt=args.dt, duration=args.duration, method=args.method, AB_order=args.AB_order, plot=args.plot, make_gif=args.make_gif, test_case=args.test_case)