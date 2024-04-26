import csv
import os
from pathlib import Path
import logging
from datetime import datetime
import re
import json


ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = Path.joinpath(ROOT_DIR, 'data')

# place your openai key in this array. You can put multiple keys if you want to run multiple threads
open_ai_keys = [""]   


def get_data_path(file: str):
    return str(Path.joinpath(ROOT_DIR, 'data', file))


def write_csv_file(file_path, header = None, data = None, overwrite=False):
    sol_path = get_data_path(file_path)
    if os.path.exists(sol_path) and overwrite is False:
        if data is not None: 
            append_csv_file(file_path=sol_path, data=data)
    else:
        with open(sol_path, "wt", encoding="UTF8", newline='') as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            if data is not None:
                writer.writerows(data)
    return sol_path
        
        
def append_csv_file(file_path, data):
    with open(file_path, "a", encoding="UTF8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

       
def init_logger(log_name, dir="logs", time_sensitive=True):
    #now we will Create and configure logger 
    if time_sensitive:
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H%M%S")
        f_name = f"{date_time}_{log_name}"
    else:
        f_name = log_name
    
    _path = get_data_path(dir)
    if not os.path.exists(_path):
        os.makedirs(_path)    
        
    handler = logging.FileHandler(get_data_path(f"{dir}/{f_name}.log"), mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger=logging.getLogger(f_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def get_optimal_value(op_file):
    sol_path = get_data_path(op_file)
    optimal_values = {}
    with open(sol_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                optimal_values[f"{row[0]}-{row[1]}-{row[2]}"] = float(row[3])
            line_count += 1
    return optimal_values
    
    
def detect_error_list(data, count, reverse=True):
    res = sorted(data, key=lambda k: (k['gap'], k['steps']), reverse=reverse)
    if len(res) > count:
        return res[:count]
    return res

def extract_edit_prompt(response):
    pattern = r'<START>\s*(.*?)\s*<END>'
    result_list = re.findall(pattern, response, re.DOTALL)
    if len(result_list) == 0:
        pattern = r'<START>(.*?)<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
    return result_list 

def load_eval_data(config):
    with open(f"{config['data_path']}{config['dataset']}/ID/test_seed_{config['seed']}.json", 'r') as json_file:
        test_data = json.load(json_file)
    return test_data


def save_text_table(table, file_name):
    p = get_data_path(f"po/{file_name}.json")
    json_object = json.dumps([ob.__dict__ for ob in table])
    with open(p, "w") as outfile:
        outfile.write(json_object)

            