import networkx
import numpy as np
import pandas as pd
from pyCombinatorial.algorithm import nearest_insertion, random_insertion, farthest_insertion, nearest_neighbour, christofides_algorithm
from pyCombinatorial.utils import graphs, util
import tsplib95
import utils


def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.int32(np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a))+np.float32(0.5)).squeeze()
                   

def solve(tsp_path, algorithm):
    p = utils.get_data_path(tsp_path)
    problem = tsplib95.load(p)
    coordinates = np.array([problem.node_coords[i] for i in problem.node_coords])
    distance_matrix = build_distance_matrix(coordinates)
    route, distance = None, -1
    if algorithm == "NI":
        route, distance = nearest_insertion_solver(distance_matrix)
    elif algorithm == "RI":
        route, distance = random_insertion_solver(distance_matrix)
    elif algorithm == "FI":
        route, distance = farthest_insertion_solver(distance_matrix)
    elif algorithm == "NN":
        route, distance = nearest_neighbour_solver(distance_matrix)
    elif algorithm == "Christofides":
        route, distance = christofides_solver(distance_matrix)
    if route is not None:
        route = route[:-1]
        route = [x-1 for x in route]
    return route, distance
    
    
def nearest_insertion_solver(dist_matrix):
    # NI - Parameters
    parameters = {
                'initial_location': 1, # -1 =  Try All Locations.
                'verbose': True
                }
    route, distance = nearest_insertion(dist_matrix, **parameters)
    return route, distance


def random_insertion_solver(dist_matrix):
    # RI - Parameters
    parameters = {
                'initial_location': 1, # -1 =  Try All Locations.
                'verbose': True
                }
    route, distance = random_insertion(dist_matrix, **parameters)
    return route, distance


def farthest_insertion_solver(dist_matrix):
    # FI - Parameters
    parameters = {
                'initial_location': 1, # -1 =  Try All Locations.
                'verbose': True
                }
    route, distance = farthest_insertion(dist_matrix, **parameters)
    return route, distance


def nearest_neighbour_solver(dist_matrix):
    # NN - Parameters
    parameters = {
                'initial_location': 1, # -1 =  Try All Locations.
                'local_search': False, 
                'verbose': True
                }
    route, distance = nearest_neighbour(dist_matrix, **parameters)
    return route, distance


def christofides_solver(dist_matrix):
    # Christofides Algorithm - Algorithm
    route, distance = christofides_algorithm(dist_matrix, local_search = True, verbose = True)
    return route, distance
