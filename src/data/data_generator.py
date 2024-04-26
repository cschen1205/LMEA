import numpy as np
import utils
import math
import tsplib95
from pyconcorde.concorde.tsp import TSPSolver
from pyconcorde.concorde.problem import Problem
from pyconcorde.concorde.util import write_tsp_file
# from pyCombinatorial.utils import util


def calculate_length(problem, trace):
    coordinates = [problem.node_coords[i] for i in problem.node_coords]
    tour = [coordinates[i] for i in trace]
    dist = 0
    for i in range(len(tour)):
        if i == len(tour)-1:
            n = 0
        else:
            n = i+1
        dist += (int)(math.sqrt(math.pow(tour[n][0]-tour[i][0], 2) + math.pow(tour[n][1]-tour[i][1], 2))+0.5)
    return dist


def concorde_solve(tsp_path:str):
    solver = TSPSolver.from_tspfile(tsp_path)
    solution = solver.solve()
    return solution


def generate_tsp(f_name: str, size: int, seed = -1, dir=None):
    if dir is not None:
        p = f"{dir}/{f_name}.tsp"
    else:
        p = f"{f_name}.tsp"
    _path = utils.get_data_path(p)
    if seed > -1:
        np.random.seed(seed)
    coordinates = np.random.randint(low=1, high=100, size=[2, size])
    Problem.from_coordinates(xs=coordinates[0], ys=coordinates[1])
    with open(_path, "wt") as fp:
        write_tsp_file(fp, xs=coordinates[0], ys=coordinates[1], norm='EUC_2D', name=f_name)   
    return _path


def generate_ctsp(f_name: str, size: int, seed = -1, dir=None):
    if dir is not None:
        p = f"{dir}/{f_name}.tsp"
    else:
        p = f"{f_name}.tsp"
    _path = utils.get_data_path(p)
    if seed > -1:
        np.random.seed(seed)
    cluster_size = (int)(size/5)
    clusters = np.random.randint(low=-50, high=50, size=[2, cluster_size])
    coordinates = np.zeros((2, size), dtype=int)
    for j in range(size):
        t = np.random.randint(cluster_size)
        coordinates[0][j] = clusters[0][t] + (int)(np.random.standard_normal()*50/math.sqrt(size)) + 50
        coordinates[1][j] = clusters[1][t] + (int)(np.random.standard_normal()*50/math.sqrt(size)) + 50
    Problem.from_coordinates(xs=coordinates[0], ys=coordinates[1])
    with open(_path, "wt") as fp:
        write_tsp_file(fp, xs=coordinates[0], ys=coordinates[1], norm='EUC_2D', name=f_name)   
    return _path
