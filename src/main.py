import os
import time
import pickle
import argparse
from utils import *
import pandas as pd
from model import Threat_Model
from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities

np.random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--numExp', type=int, default=1,
                    help='numExp')
parser.add_argument('--graph_type', type=str, default='Amazon',
                    help='graph type')
parser.add_argument('--mode', type=str, default='equalAlpha',
                    help='trade-off parameters mode')
parser.add_argument('--save_result', type=int, default=1,
                    help='whether or not to save the result to the disk')
parser.add_argument('--weighted', type=int, default=0,
                    help='whether the graph is weighted')
args = parser.parse_args()


# parameters for running experiments
learning_rate = 1
MAPPING = {
        'equalAlpha': [1/3, 1/3, 1/3],
        'alpha1=1': [1/3, 0, 0],
        'alpha2=0': [1/3, 0, 1/3],
        'alpha3=0': [1/3, 1/3, 0],
        'alpha3=1': [0, 0, 1/3],
        'alpha1=0': [0, 1/3, 1/3]
        }




## Algorithm 1 in the paper
def SGD_attack(Attacker, Optimizer):
    t1 = time.time()
    Attacker_budget = Attacker.get_budget()
    cnt = 0
    while True:
        Loss = Attacker()
        Optimizer.zero_grad()
        Loss.backward()

        budget_this_step = Attacker.get_step_budget()
        current_used_budget = Attacker.get_used_budget()
        if current_used_budget + budget_this_step <= Attacker_budget:
            Optimizer.step()
            Attacker.update_used_budget(budget_this_step)
            current_used_budget = Attacker.get_used_budget()
            cnt += 1
        else:
            break
    t_sgd = time.time() - t1
    print("SGD iterations: {}    Time: {:.4f}".format(cnt, t_sgd))
   
    ## If the graph is weighted convert all negative entries in the 
    ## adjacency matrix to zeros; otherwise call the rounding procedure.
    if args.weighted:
        Attacker.adj_tensor.data[Attacker.adj_tensor.data < 0] = 0
        addedEdges = 0
    else:
        Attacker.adj_tensor.data, addedEdges = rounding(Attacker)
    Attacker()

    lambda1_S, impact_S, centrality = Attacker.get_result()
    lambda1_S_0, impact_S_0, centrality_0 = \
            Attacker.lambda1_S_original, Attacker.impact_S_original, Attacker.centrality_original

    lambda1_S_increase_ratio =  (lambda1_S - lambda1_S_0) / lambda1_S_0
    impact_S_increase_ratio = (impact_S - impact_S_0) / torch.abs(impact_S_0)
    centrality_increase_ratio = (centrality - centrality_0) / centrality_0
    utility = Attacker.get_utility()

    return (lambda1_S_increase_ratio.detach().numpy().squeeze(),
            impact_S_increase_ratio.detach().numpy().squeeze(),
            centrality_increase_ratio.detach().numpy().squeeze(),
            utility.detach().numpy().squeeze(),
            addedEdges)


## launch attack algorithm
def launch_attach():
    alpha_1, alpha_2, alpha_3 = MAPPING[args.mode]
    Alpha = [alpha_1, alpha_2, alpha_3]
    print("alpha_1: {:.4f}      alpha_2: {:.4f}     alpha_3: {:.4f}\n".format(alpha_1, alpha_2, alpha_3))

    Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G, args.weighted)
    Optimizer = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

    t1 = time.time()
    lambda1_S_ret, impact_S_ret, centrality_ret, utility_ret, addedEdges = SGD_attack(Attacker, Optimizer)
    print("Time: {:.4f}".format(time.time() - t1))

    avgDegDiff          = Attacker.diff_avgDeg()
    avgDegDiff_S        = Attacker.diff_avgDeg(S)
    avgDegDiff_S_prime  = Attacker.diff_avgDeg(S_prime)
    adjNormDiff         = Attacker.diff_adjNorm()
    addedEdgesRatio     = addedEdges / len(G.edges())

    ## record all the statistics we are interested in
    ret = (utility_ret, lambda1_S_ret, impact_S_ret, centrality_ret, budget_change_ratio, \
        Alpha, avgDegDiff, avgDegDiff_S, avgDegDiff_S_prime, adjNormDiff, addedEdgesRatio, Attacker)
    print("Budget: {:.2f}%  \
           lambda1_S: {:.4f}%  \
           impact on S: {:.4f}% \
           centrality: {:.4f}%  \
           addedEdgesRatio: {:.4f}% \
           utility: {:.4f}\n".format(
              budget_change_ratio*100, lambda1_S_ret*100, impact_S_ret*100, centrality_ret*100, addedEdgesRatio*100, utility_ret))
    print('*' * 80)
    return ret


## the solution from the threat model is a matrix 
## with fractional entries, so we need to round it 
## to a matrix with only integral entries for unweighted graphs
def rounding(Attacker):
    A_attacked = Attacker.get_attacked_adj().numpy()
    A = Attacker.original_adj.numpy()
    idx = np.nonzero(np.triu(A_attacked, 1) != np.triu(A, 1))
    numChange = len(idx[0])

    modifiedEdges = []
    for i in range(numChange):
        rowIdx, colIdx = idx[0][i], idx[1][i]
        w_a = A_attacked[rowIdx][colIdx]
        w_o = A[rowIdx][colIdx]
        Diff = np.abs(w_a - w_o)
        AddedEdge   = (w_a > w_o and w_o == 0)
        DeletedEdge = (w_a < w_o and w_o >  0)
        if AddedEdge or DeletedEdge:
            modifiedEdges.append((rowIdx, colIdx, Diff, DeletedEdge))
    ## sorting the modifications in descending order of their magnitudes
    modifiedEdges = sorted(modifiedEdges, key=lambda x: x[2], reverse=True)

    B = A.copy()
    ## no modification made 
    if len(modifiedEdges) == 0:
        return ( torch.tensor(B, dtype=torch.float32), 0 )

    cnt = 0
    addedEdges   = 0
    deletedEdges = 0
    spec_norm = estimate_sym_specNorm(B-A)
    while (spec_norm <= Attacker.budget):
        Edge = modifiedEdges[cnt]
        ## check if "Edge" corresponds to a deletion
        DeletedEdge = Edge[-1]
        ## "Edge" is either a deletion or an addition
        if not DeletedEdge:
            B[Edge[0], Edge[1]] = 1
            B[Edge[1], Edge[0]] = 1
            addedEdges += 1
        else:
            B[Edge[0], Edge[1]] = 0
            B[Edge[1], Edge[0]] = 0
            deletedEdges += 1  

        ## if budget constraints are violated
        ## revert the operations above
        spec_norm = estimate_sym_specNorm(B-A)
        if spec_norm > Attacker.budget:
            if not DeletedEdge:
                B[Edge[0], Edge[1]] = 0
                B[Edge[1], Edge[0]] = 0
                addedEdges -= 1
            else:
                B[Edge[0], Edge[1]] = 1
                B[Edge[1], Edge[0]] = 1
                deletedEdges -= 1               
            break
        else:     
            cnt += 1
            if cnt >= len(modifiedEdges):
                break
    print("Added edges: {}      Deleted edges: {}\n".format(addedEdges, deletedEdges))
    return ( torch.tensor(B, dtype=torch.float32), addedEdges )


if __name__ == '__main__':
    result = defaultdict(list)
    graph_ret = defaultdict(list)

    for i in range(args.numExp):
        G = gen_graph(args.graph_type, i)
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        adj = nx.adjacency_matrix(G).todense()

        if args.graph_type in ['Email', 'Amazon']:
            S = select_target_subgraph(G, args.graph_type, mapping)
        else:
            S = select_target_subgraph(G, args.graph_type)
        print("---Comm size: {}    Graph size: {}---".format(len(S), len(G)))

        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)

        for budget_change_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            opt_sol = launch_attach()
            result[budget_change_ratio].append(opt_sol)

            ## attacked graphs (the last entry of opt_sol is the Attacker object)
            Attacker = opt_sol[-1]
            Adj_attacked = Attacker.get_attacked_adj()
            G_attacked = nx.from_numpy_matrix(Attacker.get_attacked_adj().numpy())
            targets = [True if i in S else False for i in range(G.order())]
            targets = {idx: {'target': targets[idx]} for idx, _ in enumerate(targets)}
            nx.set_node_attributes(G_attacked, targets)
            nx.set_node_attributes(G, targets)
            graph_ret[budget_change_ratio].append({'original': G, 'attacked': G_attacked})


    if args.save_result:
        Key = args.mode
        W = 'weighted' if args.weighted else 'unweighted'
        if not os.path.exists('../result/{}'.format(W)):
            os.mkdir('../result/{}'.format(W))

        ## save original and attacked graphs
        with open('../result/{}/{}_numExp_{}_attacked_graphs_{}.p'.format(W, args.graph_type, args.numExp, Key), 'wb') as fid:
            pickle.dump(graph_ret, fid)

        ## save statistics for plotting
        with open('../result/{}/{}_numExp_{}_ret_{}.p'.format(W, args.graph_type, args.numExp, Key), 'wb') as fid:
            pickle.dump(result, fid)



