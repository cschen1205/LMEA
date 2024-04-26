import os
import csv
import threading
import statistics 
from utils import write_csv_file, get_data_path, init_logger, get_optimal_value, open_ai_keys
from data.data_generator import generate_tsp, generate_ctsp, concorde_solve
from models.llm_tsp import evaluate
from models.tradition_tsp import solve
from numpy import mean
from scipy.stats import sem


tsp_aglorithms = ["NI", "RI", "FI", "NN", "Christofides"]
tsp_llm = ["gpt-3.5-turbo-1106"] 
exp_node_numbers = [10, 15, 20, 25]
default_problem_number = 10
default_exe_number = 1
llm_iteration_number = 450
llm_stop_step = 50


def generate_tsp_problems(tsp_name, node_nums, problem_num):
    for j in node_nums:
        for i in range(problem_num):
            generate_tsp(f_name=f"{tsp_name}_{j}_{i+1}", size=j, dir="tsp", seed=10+10*i)
            
            
def run_llm_tsp_op(problems, model, api_key, logger_name, gaps, system, state="train", iter_num=30, stop_steps=-1, record_file="rue_solution", random_select=True, device="cuda:0", 
                algorithm="ec-op", adaptive=False, start_iter=0, init_population=None, pop_num=16, prompt_type=0):
    header = ["state", "model", "node number", "problem", "distance", "optimal distance", "gap", "success_step", "route"]
    sol_path = write_csv_file(file_path=f"po/{record_file}.csv", header=header, overwrite=False)    
    iter_header = ["state", "model", "node number", "problem", "iteration", "distance", "optimal distance", "gap"]
    iter_path = write_csv_file(file_path=f"po/{record_file}_iter.csv", header=iter_header, overwrite=False)
    success_step = "NA"
    logger = init_logger(logger_name, dir="po/logs")
    print("init logger: ", logger_name)
    res = []
    for problem in problems:
        success_step = "NA"
        steps, gap, min_dist, route, points, population = evaluate(tsp_path=problem["path"], iter_log_path=iter_path, model=model, api_key=api_key, optimal_value=problem["value"], 
                                            iter_num=iter_num, logger=logger, stop_steps=stop_steps, system_msg=system, pop_num=pop_num, node_num=problem['node'], 
                                            problem_index=problem["index"], batch_index = state, random_select=random_select, device=device, algorithm=algorithm, 
                                            adaptive=adaptive,  start_iter=start_iter, init_population=init_population, prompt_type=prompt_type)
        if steps != -1:
            success_step = str(steps)
        all_population = " ".join([x[0] for x in population])
        res.append({"problem":problem['path'], "input":f"coordinates: {points}\ntraces and lengths: {all_population}", "gap": gap, "steps":steps})
        with open(sol_path, "a", encoding="UTF8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([state, model, str(problem['node']), str(problem['index']), str(min_dist), str(problem["value"]), "{:.3f}".format(gap), success_step, route])
    if type(gaps) is dict:
        gaps[system] = res
    else:
        gaps.extend(res)
    return


def run_llm_tsp(file_name, optimals, node_nums, iter_num, stop_steps, problem_index, batch_count, api_key, logger, 
                record_file="tsp_llm_solution", system = None, random_select=True, device="cuda:0", algorithm="basic", adaptive=False,
                start_iter=0, init_population=None, pop_num=16, prompt_type=0):
    out_path = get_data_path("output")
    if not os.path.exists(out_path):
        os.makedirs(out_path)    
    
    header = ["batch", "model", "node number", "problem", "distance", "optimal distance", "gap", "success_step", "route"]
    sol_path = write_csv_file(file_path=f"output/{record_file}_{problem_index}.csv", header=header, overwrite=False)    
    iter_header = ["batch", "model", "node number", "problem", "iteration", "distance", "optimal distance", "gap"]
    iter_path = write_csv_file(file_path=f"output/{record_file}_iter_{problem_index}.csv", header=iter_header, overwrite=False)
    for bt in range(batch_count):
        for model in tsp_llm:
            for j in node_nums:
                success_step = "NA"
                p = get_data_path(f"tsp/{file_name}_{j}_{problem_index}.tsp")
                op = optimals[f"{file_name}-{j}-{problem_index}"]
                steps, gap, min_dist, route, _, _ = evaluate(tsp_path=p, iter_log_path=iter_path, model=model, api_key=api_key, optimal_value=op, iter_num=iter_num, logger=logger,
                                                       stop_steps=stop_steps, system_msg=system, pop_num=pop_num, node_num=j, problem_index=problem_index, batch_index = bt+1, 
                                                       random_select=random_select, device=device, algorithm=algorithm, adaptive=adaptive,  start_iter=start_iter, 
                                                       init_population=init_population, prompt_type=prompt_type)
                if steps != -1:
                    success_step = str(steps)
                with open(sol_path, "a", encoding="UTF8", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([str(bt+1), model, str(j), str(problem_index), str(min_dist), str(op), "{:.3f}".format(gap), success_step, route])

    
def run_optimal_tsp(file_names, node_nums, problem_num):
    header = ["file", "node number", "problem", "distance", "route"]
    data = []
    for file_name in file_names:
        for j in node_nums:
            for i in range(problem_num):
                p = get_data_path(f"tsp/{file_name}_{j}_{i+1}.tsp")
                sol = concorde_solve(p)
                data.append([file_name, str(j), str(i+1), str(int(sol.optimal_value)), ",".join([str(x) for x in list(sol.tour)])])
    write_csv_file("tsp/tsp_optimal_solution.csv", header, data)
    
    
def run_traditional_tsp(file_names, optimals, node_nums, problem_num, exe_num = 1):
    header = ["batch", "file", "algorithm", "node number", "problem", "distance", "optimal distance", "gap"]
    data = []
    for file_name in file_names:
        for alg in tsp_aglorithms:
            for j in node_nums:
                for i in range(problem_num):
                    for et in range(exe_num):
                        route, distance = solve(tsp_path=f"tsp/{file_name}_{j}_{i+1}.tsp", algorithm=alg)
                        op = optimals[f"{file_name}-{j}-{i+1}"]
                        gap = "{:.3f}".format((distance - op)/op *100)
                        data.append([str(et+1), file_name, alg, str(j), str(i+1), str(int(distance)), str(op), gap])
    write_csv_file("tsp/tsp_tradition_solution.csv", header, data)


def run_llm_tsp_multithreads(file_names, prob_count, node_num, iter_num):
    optimals = get_optimal_value(op_file="tsp/tsp_optimal_solution.csv")
    keyIndex = 0
    for file_name in file_names:
        for i in range(prob_count):
            logger = init_logger(f"{file_name}{i+1}")
            logger.info(f"experiment: {file_name}-{i+1}, api key: {open_ai_keys[keyIndex]}")
            thread = threading.Thread(target=run_llm_tsp, args=(file_name, optimals, node_num, iter_num, -1, i+1, 1, open_ai_keys[keyIndex], logger, 
                                                                f"{file_name}_llm_solution_ec"))
            thread.start()
            keyIndex = keyIndex + 1 
        
        
def generate_report():
    data = {}
    model_data = None
    for i in range(5):
        sol_path = get_data_path(f"tsp/tsp_llm_solution_ec_{i+1}.csv")
        with open(sol_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    if row[1] not in data:
                        data[row[1]] = {"gap": {}, "step": {}}
                        model_data = data[row[1]]
                    if row[2] not in model_data["gap"]:
                        model_data["gap"][row[2]] = [float(row[6])]
                    else:
                        model_data["gap"][row[2]].append(float(row[6]))
                    if row[7] != "NA":
                        if row[2] not in model_data["step"]:
                            model_data["step"][row[2]] = [float(row[7])]
                        else:
                            model_data["step"][row[2]].append(float(row[7])) 
                line_count += 1
    tsp_sol_path = get_data_path("tsp/tsp_tradition_solution.csv")
    with open(tsp_sol_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if row[0] not in data:
                    data[row[0]] = {"gap": {}, "step": {}}
                    model_data = data[row[0]]
                if row[1] not in model_data["gap"]:
                    model_data["gap"][row[1]] = [float(row[5])]
                else:
                    model_data["gap"][row[1]].append(float(row[5]))
            line_count += 1
    report_data, report = {}, []      
    for m in data:
        for p in data[m]["gap"]:
            _mean = mean(data[m]["gap"][p])
            _std_error = sem(data[m]["gap"][p])
            _gap = f"{_mean:.2f} +_ {_std_error:.2f}"
            if len(data[m]["step"]) > 0 and p in data[m]["step"]:
                s_mean = mean(data[m]["step"][p])
                s_std_error = sem(data[m]["step"][p])
                s_count = len(data[m]["step"][p])
                if s_count == 1:
                    s_std_error = 0
                s_step = f"{s_mean:.2f} +_ {s_std_error:.2f} ({s_count})"
            else:
                s_step = "N/A (0)"
            if m not in report_data:
                report_data[m] = {"gap": {}, "step": {}}
            report_data[m]["gap"][p] = _gap
            report_data[m]["step"][p] = s_step
    header = ["node number", "NI gap (%)", "RI gap (%)", "Christofides gap (%)", "NN gap (%)", "FI gap (%)", "gpt-3.5-turbo gap (%)", "gpt-3.5-turbo steps (successes)"]
    for pi in ["10", "15", "20"]:
        report.append([pi, report_data["NI"]["gap"][pi], report_data["RI"]["gap"][pi], report_data["Christofides"]["gap"][pi], report_data["NN"]["gap"][pi],
                       report_data["FI"]["gap"][pi], report_data["gpt-3.5-turbo"]["gap"][pi], report_data["gpt-3.5-turbo"]["step"][pi]])
    write_csv_file(file_path="tsp/tsp_report_ec.csv", header=header, data=report)
 

# generate_tsp_problems(tsp_name="rue", node_nums=exp_node_numbers, problem_num=10)
# route, distance = solve(tsp_path="tsp/tsp_10_1.tsp", algorithm="NI")
# p = utils.get_data_path("tsp/tsp_10_1.tsp")
# problem = tsplib95.load(p, tsplib95.models.StandardProblem)
# dist = calculate_length(problem=problem, trace=route)
# print(f"route : {type(route)}, {route}, {distance}, {dist}")
# run_optimal_tsp(["rue", "clu"], node_nums=exp_node_numbers, problem_num=default_problem_number)
# optimals = get_optimal_value(op_file="tsp/tsp_optimal_solution.csv")
# run_traditional_tsp(["rue", "clu"], optimals=optimals, node_nums=exp_node_numbers, problem_num=default_problem_number, exe_num=default_exe_number)

# run_llm_tsp_multithreads(file_names=["rue100"], prob_count=default_problem_number, node_num=exp_node_numbers, iter_num=llm_iteration_number)
# generate_report()