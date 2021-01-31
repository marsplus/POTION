"""
SIS_simulations.py
------------------

Simulate SIS dynamics on graph and attacked graph.

"""
import os
import EoN
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, default='BA',
                help='graph type')
parser.add_argument('--numExp', type=int, default=1,
                help='numExp')
parser.add_argument('--numSim', type=int, default=2000,
                help='number of simulations')
parser.add_argument('--numCPU', type=int, default=4,
                help='number of CPUs to run the simulation')
parser.add_argument('--gamma', type=float, default='0.24',
                help='gamma')
parser.add_argument('--tau', type=float, default='0.2',
                help='tau')
parser.add_argument('--weighted', type=str, default='weighted',
                    help='whether the graph is weighted')
args = parser.parse_args()


GAMMA = args.gamma                       # recovery rate
TAU = args.tau                           # transmission rate
TMAX = 100                               # max. simulation time
numCPU = args.numCPU                            
numSim = args.numSim                            # number of simulations

def run_sis(original, attacked, budget, num_sim=numSim):
    """run SIS simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    for name in graphs:
        G = graphs[name]
        numNode = G.order()
        print('Simulating SIS on {}'.format(name))

        ## targeted and non-targeted subgraphs
        S = [i for i in range(numNode) if G.nodes[i]['target']]
        SP = list(set(range(numNode)) - set(S))
        for ns in range(num_sim):
            if ns % 50 == 0: print("numSim: {}".format(ns))
            
            if args.graph_type not in ['Airport', 'Protein', 'Brain']:
                sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, return_full_data=True)
            else:
                sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, transmission_weight='weight', return_full_data=True)
            ## average results over the last 30 steps
            inf_ratio_target    = np.mean([Counter(sim.get_statuses(S, i).values())['I'] / len(S) for i in range(-1, -31, -1)])
            inf_ratio_bystander = np.mean([Counter(sim.get_statuses(SP, i).values())['I'] / len(SP) for i in range(-1, -31, -1)])
            rows.append((name, inf_ratio_target, inf_ratio_bystander, budget))
    return pd.DataFrame(rows, columns=['graph', 'ratio targets', 'ratio bystanders', 'budget'])



def dispatch(params):
    Key = params
    print("Current exp: {}".format(Key))
    attackedDataPath = '../result/{}/{}_numExp_{}_attacked_graphs_{}.p'.format(args.weighted, args.graph_type, args.numExp, Key)
    with open(attackedDataPath, 'rb') as fid:
        graph_ret = pickle.load(fid)
    result = []
    for budget in [0.1, 0.2, 0.3, 0.4, 0.5]:
        graph_param = graph_ret[budget]
        for item in graph_param:
            original, attacked = item['original'], item['attacked']
            ret = run_sis(original, attacked, budget)
            result.append(ret)
    result = pd.concat(result)
    return result


"""Simulate dynamics on the original and attacked graphs."""
pool = Pool(processes=numCPU)
params = []
expName = ['equalAlpha']
for exp in expName:
    params.append(exp)

ret = pool.map(dispatch, params)
folder = '../result/{}/{}-SIS/Gamma-{:.2f}---Tau-{:.2f}/'.format(args.weighted, args.graph_type, GAMMA, TAU)
if not os.path.exists(folder):
    os.makedirs(folder)

for idx, exp in enumerate(expName):
    result = ret[idx]
    fileName = '{}_numExp_{}_SIS_{}.p'.format(args.graph_type, args.numExp, exp) 
    with open(os.path.join(folder, fileName), 'wb') as fid:
        pickle.dump(result, fid)
pool.close()
pool.join()
