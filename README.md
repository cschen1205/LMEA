# LMEA
Code for paper: [Large Language Models as Evolutionary Optimizers](https://arxiv.org/abs/2310.19046)


# Quick Start

## Setup Environment
```bash
# clone the repo
git clone https://github.com/cschen1205/LMEA.git

# Go to directory
cd LEMA

# create a new environment and activated it
python -m venv venv
source ./venv/bin/activate

# pyconcorde installation and steup
pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

# install dependent pacakges
pip install -r requirements.txt
```

## Setup OpenAI Key

In the **src/utils.py** file, line 14, place the openai keys in array "**open_ai_keys**"

```bash
# place your openai key in this array. You can put multiple keys if you want to run multiple threads
open_ai_keys = [""]   
```

## Dataset

The TSP problems used in experiments of this page are located in Folder **data/tsp**

# Generate TSP problem files

run src/problem.py to generate TSP problems files

```bash
python scr/problem.py -name "tsp" -t "rue" -d "tsp" -nc 10,15 -pc 10
```


## Run experiment

You can run the **experiment.py** python script for experiments. For example:

```python
python src/executor.py -n rue,clu -nc 10,15 -pi 1 -al ec
```

The above command will run "**LLM-EC**" experiments on the these problem files:

```
clu_10_1.tsp (clu type, 10 nodes, 1st problem instance)
clu_15_1.tsp (clu type, 15 nodes, 1st problem instance)
rue_10_1.tsp (rue type, 10 nodes, 1st problem instance)
rue_15_1.tsp (rue type, 15 nodes, 1st problem instance)
```

The log files of the experiments will be saved in folder "**data/logs**" 