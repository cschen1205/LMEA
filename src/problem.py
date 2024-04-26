import argparse
import os
import utils
from pyconcorde.concorde.tsp import TSPSolver
from data.data_generator import generate_tsp, generate_ctsp

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", default="rue", help="problem name", type=str)
argParser.add_argument("-t", "--type", default="rue", type=str, help="[rue|clu]")
argParser.add_argument("-nc", "--node_count", default=[10],  type=list_of_ints)
argParser.add_argument("-pc", "--problem_count", default=10, type=int)
argParser.add_argument("-d", "--dir", default="tsp", type=str)
argParser.add_argument("-s", "--seed", default=10, type=int)

args = argParser.parse_args()
if args.dir is not None:
    _path = utils.get_data_path(args.dir)
    if not os.path.exists(_path):
        os.makedirs(_path)


def generate_tsp_problems(tsp_name, node_nums, problem_num, seed=10, type="rue", dir=None):
    for j in node_nums:
        for i in range(problem_num):
            if type == "rue":
                generate_tsp(f_name=f"{tsp_name}_{j}_{i+1}", size=j, dir=dir, seed=seed+10*i)
            else:
                generate_ctsp(f_name=f"{tsp_name}_{j}_{i+1}", size=j, dir=dir, seed=seed+10*i)
            
               
generate_tsp_problems(tsp_name=args.name, node_nums=args.node_count, problem_num=args.problem_count, 
                      seed=args.seed, type=args.type, dir=args.dir)