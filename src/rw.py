import os
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, default='Email',
                help='graph type')
parser.add_argument('--numExp', type=int, default=1,
                help='numExp')
parser.add_argument('--weighted', type=str, default='unweighted',
                    help='whether the graph is weighted')
parser.add_argument('--numSim', type=int, default=100,
                    help='number of simulations')
parser.add_argument('--algo', type=str, default='RWR',
                    help='which random-walk based algorithm to run')
parser.add_argument('--restart', type=float, default=0.1,
                    help='restart prob.')
parser.add_argument('--tol', type=float, default=1e-8,
                    help='termination tolerance')
args = parser.parse_args()

np.random.seed(123)

def random_walk_algo(original, attacked, budget, algo='rwr', c=0.1, numSim=100, maxIter=100, eps=1e-8):
    """
    simulate random-walk based algorithm, e.g., random walk with restart, PageRank, etc.
    c: re-start probability
    """
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    numNodes = graphs['original'].order()

    ## targeted and non-targeted subgraphs
    S = [i for i in range(numNodes) if graphs['attacked'].nodes[i]['target']]
    SP = list(set(range(numNodes)) - set(S))

    for name in graphs:
        G = graphs[name]
        Adj = nx.adjacency_matrix(G).todense()
        D_norm = np.diag( 1 / (Adj * np.ones((numNodes, 1))).A1 )
        ## row normalize the adj. matrix
        Adj = D_norm * Adj

        ## In each iteration we randomly 
        ## pick a node from the non-targeted subgraph
        avgDis_S_ret  = []
        avgDis_SP_ret = []
        for sim in range(numSim):
            if sim % 50 == 0:
                print("Number of simulation: ", sim)

            ## randomly pick a starting node
            if algo == 'RWR':
                ns = np.random.choice(SP)
            elif algo == 'PR':
                ns = np.random.choice(range(numNodes))
            v = np.zeros((numNodes, 1))
            v[ns] = 1

            r0 = v.copy()
            all_one = np.ones((numNodes, 1))
            for k in range(maxIter):
                if algo == 'RWR':
                    r = (1 - c) * Adj.T * r0 + c * v
                elif algo == 'PR':
                    r = (1 - c) * Adj.T * r0 + c * all_one / numNodes
                else:
                    raise ValueError("Unknown random-walk algorithm")

                Diff = np.linalg.norm(r0-r, 2)
                if Diff < eps:
                    break
                else:
                    r0 = r
            
            avgDis_S_ret.append(np.sum(r[S]))
            avgDis_SP_ret.append(np.sum(r[SP]))

        avgDis_S = np.mean(avgDis_S_ret)
        avgDis_S_std = np.std(avgDis_S_ret)
        avgDis_SP = np.mean(avgDis_SP_ret)
        avgDis_SP_std = np.std(avgDis_SP_ret)   
        rows.append((avgDis_S, avgDis_S_std, avgDis_SP, avgDis_SP_std, budget, name))

    return pd.DataFrame(rows, columns=['avgDis_S', 'avgDis_S_std', 'avgDis_SP', 'avgDis_SP_std', \
                                       'budget', 'graph'])


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
            np.random.seed(9)
            original, attacked = item['original'], item['attacked']
            ret = random_walk_algo(original, attacked, algo=args.algo, budget=budget, \
                                   c=args.restart, numSim=args.numSim, eps=args.tol)
            result.append(ret)
    result = pd.concat(result)
    return result



"""Simulate dynamics on the original and attacked graphs."""
numCPU = 1 
pool = Pool(processes=numCPU)
params = []
#expName = ['equalAlpha', 'alpha1=1', 'alpha2=0', 'alpha3=1']
expName = ['equalAlpha']
for exp in expName:
    params.append(exp)

ret = pool.map(dispatch, params)
folder = '../result/{}/random-walk/'.format(args.weighted)
if not os.path.exists(folder):
    os.makedirs(folder)

for idx, exp in enumerate(expName):
    result = ret[idx]
    fileName = '{}_{}_{}.p'.format(args.graph_type, args.algo, exp) 
    with open(os.path.join(folder, fileName), 'wb') as fid:
        pickle.dump(result, fid)
pool.close()
pool.join()
