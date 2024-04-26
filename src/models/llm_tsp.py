import random
import tsplib95
import time
import csv
from utils import get_data_path, append_csv_file
from data.data_generator import calculate_length
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from datetime import datetime
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from statistics import mean 


def problem_to_points(problem):
    res = ''
    for i in problem.node_coords:
        res += f"{i-1}:({problem.node_coords[i][0]},{problem.node_coords[i][1]}),"
    res = res[:-1]
    return res


def get_solutions(content:str, node_count:int, logger):
    sols = []
    idx_start = content.find("<trace>")
    while idx_start > -1:
        idx_end = content.find("</trace>")
        str_trace = content[idx_start+7:idx_end]
        str_trace = str_trace.replace(" ", "")
        t = [int(x) for x in str_trace.split(",") if x.isdigit()]
        t = list(dict.fromkeys(t))
        if len(t) != node_count or any(item < 0 or item >= node_count for item in t):
            t = fixed_error_solution(t, range(node_count))
            logger.debug(f"error list: {t}")
        # if len(t) == node_count and all(item > -1 and item < node_count for item in t):
        if all(t != x for x in sols):
            sols.append(t)
        content = content[idx_end+7:]
        idx_start = content.find("<trace>")
    return sols


def fixed_error_solution(input, source):
    t = set(input)
    rest = [x for x in source if x not in t]
    random.shuffle(rest)
    res = [x for x in input if x in source and x in t]
    res.extend(rest)
    return res
    
    
def evl_process(problem, sols):
    traces = []
    for i in range(len(sols)):
        dist = calculate_length(problem=problem, trace=sols[i])
        traces.append((f"<trace>{','.join([str(x) for x in sols[i]])}</trace>,length:{dist};", dist))
    traces.sort(key=lambda x: x[1], reverse=True)
    return traces


def populate(problem, pop_num):
    node_count = len(problem.node_coords)
    p = []
    for i in range(pop_num*6):
        tour = [x for x in range(node_count)]
        random.shuffle(tour)
        dist = calculate_length(problem=problem, trace=tour)
        p.append((f"<trace>{','.join([str(x) for x in tour])}</trace>,length:{dist};", dist))
    p.sort(key=lambda x: x[1], reverse=True)
    p = p[-pop_num:]
    return p


def llm_process(llm, tokens, secs, points, population, logger):
    content = " ".join([x[0] for x in population])
    logger.info(f"content: {content}")
    system_message = SystemMessage(
        content=(
            '''You are given a list of points with coordinates below: {points}.
            Below are some previous traces and their lengths. The traces are arranged in descending order based on their lengths, where lower values are better.\n
            {content}
            Give me some new traces that are different from all traces above, and each trace has a length lower than any of the above. Each trace should traverse all points exactly once.
            The trace should start with <trace> and end with </trace>.'''
        )
    )
    template = ChatPromptTemplate.from_messages(
        [system_message],
    )

    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62-secs)
        res = chain.run(points=points, content=content)
    logger.info(f"res: {res}")
    return res, tokens


def llm_process_samples(llm, tokens, secs, points, population):
    content = "\n".join([x[0] for x in population])
    system_message = SystemMessage(
        content=(
            f"You are given a list of points with coordinates below: {points}. Try to find a shortest possible trace that traverse each point exactly once and returns to the start point."
            "The distance between two points A, B equal to math.sqrt((A[0] - B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1]))."
            "And the length of a trace is the sum of all the distance of adjacent points in the trace. "
            "For example, given a list of points with coordinates: "
            "0:(441,126),1:(584,619),2:(782,170),3:(765,573),4:(493,221),5:(67,871),6:(713,614),7:(991,305),8:(875,29),9:(212,945)"
            "The shortest possible trace is <trace>0,2,8,7,3,6,1,9,5,4</trace>. And its length is 2901.0."
            "You will be given some previous traces and their lengths as input. The traces are arranged in descending order based on their lengths, where lower values are better.\n"
            "Give me some new traces that are different from all input traces, and has a length lower than any of the above. The trace should traverse all points exactly once."
            "The trace should start with <trace> and end with </trace>."
        )
    )
    template = ChatPromptTemplate.from_messages(
        [system_message, 
         HumanMessagePromptTemplate.from_template(
            f"Input: {content}"
         )]
    )

    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62-secs)
        res = chain.run(points=points, content=content)
    return res, tokens


def get_cm_prompts(prompt_type):
    crossover_msg1 = "There are 2 different crossover operators you can use: PMX (Partially Mapped Crossover), OX (Ordered Crossover)"
    mutation_msg1 = "There are 3 different mutation operators you can use: Swap Mutation, Insert Mutation, Inversion Mutation"
    crossover_msg2 = '''There are 2 different crossover operators you can use:
        1. PMX (Partially Mapped Crossover):
            - Description: PMX randomly selects a segment from parent 1, copies it to the offspring, and fills in the remaining positions of the offspring by mapping elements from parent 2.
            Below is an example.
                - Parent 1: 1 2 3 4 5 6 7 8
                - Parent 2: 3 7 5 1 6 8 2 4
                - Randomly select a segment from parent 1 (e.g., positions 4 to 6): 4 5 6
                - Copy the segment from Parent 1 to offspring solution: _ _ _ 4 5 6 _ _ 
                - Fill in the remaining positions by mapping elements from parent 2 (note elements cannot be repeated) to the offspring: 3 7 8 4 5 6 2 1
        2. OX (Ordered Crossover):
            - Description: OX randomly selects a segment from parent 1, copies it to the offspring, and fills in the remaining positions with the missing elements in the order in which they appear in parent 2.
            Below is an example.
                - Parent 1: 1 2 3 4 5 6 7 8
                - Parent 2: 3 7 5 1 6 8 2 4
                - Randomly select a segment from parent 1 (e.g., positions 4 to 6): 4 5 6
                - Copy the segment from Parent 1 to the offspring: _ _ _ 4 5 6 _ _ 
                - The missing elements in the order in which they appear in parent 2 are {3, 7, 1, 8, 2}
                - Fill in the remaining positions of the offspring based on the above sorted elements: 3 7 1 4 5 6 8 2'''
    mutation_msg2 = '''There are 3 different mutation operators you can use:
        1. Swap Mutation:
            - Description: swap mutation randomly selects two positions in an individual and swaps the elements at those two positions.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select two positions, e.g., position 3 and posision 6
                - Swap the elements 8 and 7 at position 3 and position 6: 5 2 7 4 1 8 6 3
        2. Insert Mutation:
            - Description: insert mutation randomly selects one position in the individual and moves the element at that position to another randomly chosen position.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select one position, e.g., position 3
                - Move the element 8 at position 3 to another randomly chosen position 6: 5 2 4 1 7 8 6 3
        3. Inversion Mutation:
            - Description: inversion mutation randomly selects two positions in an individual and inverts the order of the elements between those positions.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select two positions, e.g., position 3 and posision 6
                - inverts the order of the elements between position 3 and position 6: 5 2 7 1 4 8 6 3'''
    crossover_selection = "Select one of the crossover operators based on above EC knowledge , use the selected crossover operator to crossover"
    mutation_selection = "Select one of the Mutation operators based on above EC knowledge, use the selected crossover operator to mutate"   
    
    if prompt_type == 0:
        return crossover_msg1, crossover_selection, mutation_msg1, mutation_selection
    elif prompt_type == 1:
        return crossover_msg2, crossover_selection, mutation_msg2, mutation_selection
    elif prompt_type == 2:
        return crossover_msg2, crossover_selection, "", "Mutate"
    elif prompt_type == 3:
        return "", "Crossover", mutation_msg2, mutation_selection


def llm_process_ec(llm, tokens, secs, points, population, hints, logger, prompt_type = 0, random_select = False):
    all_population = " ".join([x[0] for x in population])
    self_hints = ""
    if len(hints) > 0:
        self_hints = "Here are the hints you should follow to choose corssover operator and mutation operator:" + "\n".join([x for x in hints])
    logger.info(f"content: {all_population}, hint: {self_hints}")
    
    crossover, crossover_selection, mutation, mutation_selection = get_cm_prompts(prompt_type=prompt_type)

    system_message = SystemMessage(content=('''You are an evolutionary computing expert for the Traveling Salesman Problem.
        You are given a list of points with coordinates, some traces and their lengths. 
        The traces are arranged in descending order based on their lengths, where lower values are better.
        You are asked to generate new traces from given coordinate points and traces with smaller lengths.

        For example, given the following input:
        -----START OF EXAMPLE INPUT-----
        coordinates: 0:(10,41),1:(16,37),2:(65,17),3:(1,79),4:(29,12),5:(90,55),6:(94,89),7:(30,63)
        iteration number: 2
        traces and lengths: <trace>0,1,2,3,4,5,6,7</trace>,length:430; <trace>2,6,4,0,5,7,1,3</trace>,length:520;
        -----END OF EXAMPLE INPUT-----
        EC knowledge: {crossover}\n{mutation}\n
        
        You should follow the below instruction step-by-step to generate new traces from given coordinate points and traces. 
        {self_hints}
        Ensure you preserve selected corssover operator in Step 2, selected mutation operator in Step 3, and the traces at each step, repeat Step 1, 2, 3 for a given iteration number.
        1. choose any two traces from the given traces, and save the two choosen traces, bracketed them with <sel> and </sel>.
        2. {crossover_selection} the two traces got in Step 1 and generate a new trace that is different from all traces, and has a length lower than any of these two traces. 
        The generated trace should traverse all points exactly once. Save the selected crossover operator and bracketed it with <c> and </c>. Save the generated trace and bracketed it with <cross> and </cross>.
        3. {mutation_selection} the trace generated in Step 2 and generate a new trace that is different from all traces, and has a lower length.
        The trace should traverse all points exactly once. Save the selected mutation operator and bracketed it with <m> and </m>. Save the generated trace and bracketed it with <trace> and </trace>.
        
        Directly give me all the saved selected crossover operator from Step 2, the mutation operator from Step 3, and the traces from each Step without any explanations.
        The output format should be similiar with below, and the output should contain 16 iterations:
        Iteration 1:
        Step 1: <sel>0,1,2,3,4,5,6,7</sel>, <sel>2,6,4,0,5,7,1,3</sel>
        Step 2: <c>PMX (Partially Mapped Crossover)</c><cross>2,6,7,3,4,5,1,0</cross>
        Step 3: <m>Swap Mutation</m><trace>2,6,5,3,4,7,1,0</trace>
        Iteration 2:
        Step 1: <sel>2,6,4,0,5,7,1,3</sel>, <sel>0,1,2,3,4,5,6,7</sel>
        Step 2: <c>OX (Ordered Crossover)</c><cross>2,6,0,3,4,5,7,1</cross>
        Step 3: <m>Inversion Mutation</m><trace>2,6,5,4,3,0,7,1</trace>
        '''))
        
    template = ChatPromptTemplate.from_messages(
        [system_message, HumanMessagePromptTemplate.from_template(
            '''
            coordinates: {points}
            iteration number: 16
            traces and lengths: {all_population}
            '''
        )]
    )
        
    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62-secs)
        res = chain.run(points=points, all_population = all_population, crossover=crossover, mutation=mutation,
                        crossover_selection=crossover_selection, mutation_selection=mutation_selection, self_hints=self_hints)
    logger.info(f"res: {res}")
    return res, tokens


def llm_process_ec_po(llm, tokens, secs, system, points, population, logger):
    all_population = " ".join([x[0] for x in population])
    logger.info(f"content: {all_population}")
    template = ChatPromptTemplate.from_messages(
        [SystemMessage(content=(system)), HumanMessagePromptTemplate.from_template(
            '''
            coordinates: {points}
            traces and lengths: {all_population}
            '''
        )]
    )
        
    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62-secs)
        res = chain.run(points=points, all_population = all_population)
        retry_count = 0
        while retry_count < 3 and get_iteration_num(res) < 16:
            res += "\n" + chain.run(points=points, all_population = all_population)
            retry_count += 1
    logger.info(f"res: {res}")
    return res, tokens


def get_iteration_num(res):
    return res.count("<trace>")


def self_reflection(llm, analysis, logger):
    anres, tlist = {}, ["PMX", "OX", "Swap Mutation", "Insert Mutation", "Inversion Mutation"]
    for tt in tlist:
        if tt in analysis:
            w_im = {analysis[tt]['t']} - {analysis[tt]['im']}
            anres[tt] = f"selected a total of {analysis[tt]['t']} times, with {analysis[tt]['im']} times showing improvement and {w_im} times without improvement; the average improvement in the newly generated solutions is {analysis[tt]['av']}%."
        else:
            anres[tt] = ""
    system_message = SystemMessage(
        content=(
            '''You are an advanced evolutionary optimizer for the Traveling Salesman Problem (TSP). You have gone through many iterations before to optimize a TSP instance. 
            In each iteration you select one crossover operator from PMX (Partially Mapped Crossover) and OX (Ordered Crossover), and select one mutation operator from Swap Mutation, 
            Insert Mutation, and Inversion Mutation, to generate a new solution. Your selection records are as follows.
            PMX: {pmx} 
            OX: {ox}
            
            Swap Mutation: {sm}
            Insert Mutation: {im}
            Inversion Mutation: {inver}
            
            As you will move forward to the next iteration, now your goal is to write a few sentences describing what worked so far. 
            You will need this as a hint when you select crossover and mutation in the next iteration.'''
        )
    )
    template = ChatPromptTemplate.from_messages(
        [system_message]
    )
    chain = LLMChain(llm=llm, prompt=template)
    res = chain.run(pmx=anres["PMX"], ox=anres["OX"], sm=anres["Swap Mutation"], im=anres["Insert Mutation"], inver=anres["Inversion Mutation"])
    logger.info(f"reflection: {res}")
    return res


def analysis_result(content, node_count, problem, logger, ec_type={}):
    iters = content.split("Iteration ")
    item_types = ["sel", "sel", "c", "cross", "m", "trace"]
    for iter in iters:
        if len(iter) > 0:
            sol, c, m = [], "", ""
            for tt in item_types:
                idx_start = content.find(f"<{tt}>")
                idx_end = content.find(f"</{tt}>")
                str_trace = content[idx_start+len(tt)+2:idx_end]
                if tt == "c":
                    c = str_trace
                elif tt == "m":
                    m = str_trace
                else:
                    str_trace = str_trace.replace(" ", "")
                    t = [int(x) for x in str_trace.split(",") if x.isdigit()]
                    t = list(dict.fromkeys(t))
                    if len(t) != node_count or any(item < 0 or item >= node_count for item in t):
                        t = fixed_error_solution(t, range(node_count))
                        logger.debug(f"error list: {t}")
                    dist = calculate_length(problem=problem, trace=t)
                    sol.append(float(dist))
                content = content[idx_end+len(tt)+2:]
            if len(sol) > 3:
                c_t, m_t = "PMX", "Swap Mutation"
                if "Partially Mapped Crossover".lower() in c.lower() or "pmx" in c.lower():
                    c_t = "PMX"
                else:
                    c_t = "OX"
                if c_t not in ec_type:
                    ec_type[c_t] = {"t":0, "im":0, "iml":[]}
                ec_type[c_t]["t"] += 1 
                if sol[0] > sol[2] and sol[1] > sol[2]:
                    ci = ((sol[0] - sol[2])/sol[0] + (sol[1] - sol[2])/sol[1])*50
                    ec_type[c_t]["im"] += 1
                    ec_type[c_t]["iml"].append(ci)
                if "Swap Mutation".lower() in m.lower():
                    m_t = "Swap Mutation"
                elif "Inversion Mutation".lower() in m.lower():
                    m_t = "Inversion Mutation"
                else:
                    m_t = "Insert Mutation"
                if m_t not in ec_type:
                    ec_type[m_t] = {"t":0, "im":0, "iml":[]}
                ec_type[m_t]["t"] += 1
                if sol[2] > sol[3]:
                    mi = ((sol[2] - sol[3])/sol[2])*100
                    ec_type[m_t]["im"] += 1
                    ec_type[m_t]["iml"].append(mi)
    for tt in ec_type:
        if len(ec_type[tt]['iml']) > 0:
            ec_type[tt]['av'] = mean(ec_type[tt]['iml'])
        else:
            ec_type[tt]['av'] = 0
    return ec_type

    
def evaluate(tsp_path, iter_log_path, model, api_key, optimal_value, node_num, problem_index, logger, iter_num=10, pop_num = 5, stop_steps = -1, system_msg=None,
             batch_index=1, random_select = True, algorithm="basic", device="cuda:0", adaptive=False,  start_iter=0, init_population=None, prompt_type=0):
    problem = tsplib95.load(tsp_path, tsplib95.models.StandardProblem)
    points = problem_to_points(problem)
    success_step = -1
    min_dist = float("inf")
    llm = ChatOpenAI(model=model, temperature=1.0, openai_api_key=api_key)
    # reflect_llm = ChatOpenAI(model=model, temperature=1.0, openai_api_key=api_key)
    start_time = datetime.now()
    hints = [] #, analysis = {}
    short_routes, route, worse_iter, tokens, gap, iter_result = [], "", 0, 0, float('inf'), []
    logger.debug(f"<<<<==========start LLM tsp => {tsp_path}, {model}, {iter_num}, {pop_num}, {stop_steps}==========>>>>")
    if init_population != None:
        short_routes = init_population
    for i in range(start_iter, iter_num):
        logger.debug(f"<==========iteration: {i}, {model}, {tsp_path}==========>")
        if i == 0:
            init_population = populate(problem=problem, pop_num=pop_num)
            short_routes.extend(init_population)
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
            start_time = datetime.now()
        if algorithm == "basic":
            res, tokens = llm_process(llm=llm, tokens=tokens, secs=secs, points=points, population=short_routes, logger=logger)
        elif algorithm == "ec":
            res, tokens = llm_process_ec(llm=llm, tokens=tokens, secs=secs, points=points, population=short_routes, hints=hints, random_select=random_select,
                                         logger=logger, prompt_type=prompt_type)
        else:
            res, tokens = llm_process_ec_po(llm=llm, tokens=tokens, secs=secs, system=system_msg, points=points, population=short_routes, logger=logger)
        sols = get_solutions(res, node_count=len(problem.node_coords), logger=logger)
        # analysis = analysis_result(res, node_count=len(problem.node_coords), problem=problem, logger=logger, ec_type=analysis)
        if len(sols) != 0:
            traces = evl_process(problem=problem, sols=sols)
            if len(traces) > 0: 
                if traces[-1][1] >= min_dist:
                    worse_iter += 1
                else:
                    worse_iter = 0
            if algorithm == "ec" and adaptive and worse_iter > 20:
                worse_iter = 0
                if llm.temperature < 1.2:
                    llm.temperature = llm.temperature+0.05
                    pop_num += 2
                    logger.debug(f"adaptive update: {llm.temperature}")
            traces = list(filter(lambda x: all(item[0]!=x[0] for item in short_routes), traces))
            short_routes.extend(traces)
            short_routes.sort(key=lambda x: x[1], reverse=True)
            if len(short_routes) > pop_num:
                short_routes = short_routes[-pop_num:]
                
            # short_routes = update_short_traces(short_routes, traces, pop_num, problem, worse_iter, logger)
            min_dist = short_routes[-1][1]
            route = short_routes[-1][0]
            gap = ((min_dist - optimal_value)/optimal_value)*100
            iter_result.append([str(batch_index), model, str(node_num), str(problem_index), str(i+1), str(min_dist), str(optimal_value), "{:.3f}".format(gap)])
            if len(iter_result) >= 10:
                append_csv_file(iter_log_path, iter_result)
                iter_result.clear()
            logger.debug(f"iteration: {i} {tsp_path}, optimal: {optimal_value}, gap: {(min_dist - optimal_value)/optimal_value*100}, route: {route}, worse iter: {worse_iter}")
            if min_dist <= optimal_value:
                success_step = i+1
                break
            if stop_steps > 0 and worse_iter >= stop_steps:
                break
        time.sleep(0.4)
        # if i % 2 == 0:
        #     h = self_reflection(llm=reflect_llm, analysis=analysis, logger=logger)
        #     hints = [h]
        #     analysis = {}
    if route != "":
        idx_start = route.find("<trace>")
        idx_end = route.find("</trace>")
        route = route[idx_start+7:idx_end]
    if len(iter_result) > 0:
        append_csv_file(iter_log_path, iter_result)
        iter_result.clear()   
    logger.debug(f"<==========iteration: {i} end: {model}, {tsp_path}, {optimal_value}, {success_step}, {gap}, {min_dist}, {route}==========>")    
    return success_step, gap, min_dist, route, points, init_population


def tournament_selection(population, population_size, tournament_size = 2):
    res = []
    for j in range(population_size):
        parents = []
        while len(parents) < tournament_size:
            selected_parent = None
            lowest_length = float("inf")
            for n in range(tournament_size):
                i = random.randint(0, population_size - 1)
                parent = population[i]
                if parent[1] <= lowest_length:
                    selected_parent = parent
                    lowest_length = parent[1]
            if selected_parent not in parents:
                parents.append(selected_parent)
        res.append(parents)
    return res


def random_selection(population, population_size, tournament_size = 2):
    res = []
    for j in range(population_size):
        parents = []
        while len(parents) < tournament_size:
            i = random.randint(0, population_size - 1)
            selected_parent = population[i]
            if selected_parent not in parents:
                parents.append(selected_parent)
        res.append(parents)
    return res


def update_short_traces(short_traces, add_traces, num, problem, worse_iter, logger):
    short_traces.extend(add_traces)
    short_traces.sort(key=lambda x: x[1], reverse=True)
    if len(short_traces) > num:
        short_traces = short_traces[-num:]
    
    if num > 4 and worse_iter > 5:
        k = num // 4
        pop_list = populate(problem=problem, pop_num=k)
        short_traces.extend(pop_list)
        short_traces.sort(key=lambda x: x[1], reverse=True)
        all_population = " ".join([x[0] for x in short_traces])
        logger.debug(f"update short traces: {all_population}")
    return short_traces
