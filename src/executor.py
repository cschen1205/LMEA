import argparse
import os
from experiment import run_llm_tsp, get_optimal_value, open_ai_keys, run_optimal_tsp, run_traditional_tsp
from utils import init_logger, get_data_path

file_names = ["rue", "clu"]
algorithms = ["basic", "ec"]
batch_number = 1
llm_iteration_number = 250

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_strings(arg):
    return arg.split(',')

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", help="[rue|clu]", type=list_of_strings)
argParser.add_argument("-d", "--device", default="cuda:0", type=str)
argParser.add_argument("-nc", "--node_count", default=[10], type=list_of_ints)
argParser.add_argument("-pc", "--problem_count", default=10, type=int)
argParser.add_argument("-pi", "--problem_index", default=1, type=int)
argParser.add_argument("-ki", "--key_index", default=0, type=int)
argParser.add_argument("-al", "--algorithm", default="basic", help="basic | ec", type=str)
argParser.add_argument("-ad", "--adaptive", default=True, type=bool)
argParser.add_argument("-it", "--iteration", default=0, type=bool)

args = argParser.parse_args()
pi = args.problem_index
al = args.algorithm

optimal_path = get_data_path("tsp/tsp_optimal_solution.csv")
if not os.path.exists(optimal_path):
    run_optimal_tsp(["rue", "clu"], node_nums=args.node_count, problem_num=args.problem_count)
    optimals = get_optimal_value(op_file="tsp/tsp_optimal_solution.csv")
    run_traditional_tsp(["rue", "clu"], optimals=optimals, node_nums=args.node_count, 
                        problem_num=args.problem_count, exe_num=1)
else:
    optimals = get_optimal_value(op_file="tsp/tsp_optimal_solution.csv")

for file_name in args.name:
    logger = init_logger(f"{file_name}{pi}_{al}")
    logger.info(f"experiment: {file_name}-{al}-{pi}, api key index: {args.key_index}, adaptive: {args.adaptive}")
    run_llm_tsp(file_name=file_name, optimals=optimals, node_nums=args.node_count, iter_num=llm_iteration_number, stop_steps=-1,
                problem_index=pi, batch_count=batch_number, api_key=open_ai_keys[args.key_index], logger=logger, 
                record_file=f"{file_name}_{al}_llm_solution", algorithm=al, adaptive=args.adaptive)