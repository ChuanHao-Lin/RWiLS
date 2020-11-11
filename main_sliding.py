import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import networkx as nx
from time import time
# local
from utility.aggregation import *
from utility.util import *
from utility.random_graph import *
from utility.sliding import *
# dirichlet
from dirichlet import dirichlet

# embed
from gae.embedding import GAE
from DeepWalk.embedding import DeepWalk


########## PARAMETER SETTING ##########

window_size = 10
dim = 32

a = 100 # aggregation lower bound
b = 1000 # aggregation upper bound

# True for parameter change. False for structure change
change_type = True

pairwise_transition = inner_transition # distance_transition inner_transition
estimate_method = "fixedpoint" # meanprecision fixedpoint
activate = ReLu


########## RANDOM GRAPH PARAMETERS ##########

num_nodes = 1000
block_size = 5
degree = 10.


########## PROBLEM SETTING ##########

change_period = 1

start_deg = 10.
end_deg = 10.01
start_block_size = 4
end_block_size = 5

start_period = 20
end_period = 20
pause_period = 1
epoch = 200


########## PARAMETER CHANGE ##########

if change_type:
    degs = [start_deg]
    periods = [start_period]

    for i in range(1, change_period):
        degs.append(start_deg + i * (end_deg - start_deg) / change_period)
        periods.append(pause_period)

    degs.append(end_deg)
    periods.append(end_period)

    assert len(periods) == len(degs)

    graph_model = SBM_parameter_random_graph(num_nodes, block_size, start_deg)

    num_period = len(degs)

    parameters = zip(degs)


########## STRUCTURE CHANGE ##########

else:
    reverse = (end_block_size - start_block_size < 0)

    if reverse:
        start_block_size, end_block_size = end_block_size, start_block_size

    delta = end_block_size - start_block_size

    graph_model = SBM_structure_random_graph(num_nodes, start_block_size, degree)
    start_block = graph_model.blocks.copy()
    start_prob = graph_model.prob_matrix.copy()
    start_nodelist = graph_model.nodelist.copy()

    graph_model.update(block_size=end_block_size)
    end_block = graph_model.blocks.copy()

    block_sizes = [start_block_size]
    blocks = [start_block]
    probs = [start_prob]
    nodelists = [start_nodelist]

    periods = list()
    if reverse:
        periods = [end_period]
    else:
        periods = [start_period]

    new_block = start_block + [0] * delta

    new_prob = np.pad(start_prob, (0, delta), mode="constant")
    for i in range(1, change_period + 1):
        temp_block = [j + (k - j) * i // change_period for j, k in zip(new_block, end_block)]
        # Fix round off. Only < will happen.
        if sum(temp_block) < num_nodes:
            for _ in range(num_nodes - sum(temp_block)):
                distance_min = 1
                distance_idx = end_block_size
                for j in range(1, delta + 1):
                    if temp_block[-j] / end_block[-j] < distance_min:
                        distance_min = temp_block[-j] / end_block[-j]
                        distance_idx = -j
                temp_block[distance_idx] += 1

        temp_prob = new_prob.copy()
        # Shallow copy only copy the 1st dimension. Dig down to 2nd dimension to implement deep copy.
        # [list()] * n will make all new list as reference. Loop to generate different list().
        temp_nodelist = [j.copy() for j in nodelists[-1]] + [list() for _ in range(end_block_size - len(nodelists[-1]))]

        for j in range(start_block_size):
            # No nodes
            if sum(temp_block[-delta:]) > 0:
                avg_prob = (degree - sum(temp_prob[j] * temp_block) + temp_prob[j][j]) / sum(temp_block[-delta:])
            else:
                avg_prob = 0
            temp_prob[j][-delta:] = avg_prob
            temp_prob[-delta:, j] = avg_prob

            move_node = blocks[-1][j] - temp_block[j]
            for k in range(move_node):
                found = False
                for l in range(1, delta + 1):
                    if sum(temp_block[-delta:]) > 0:
                        if sum(temp_block[-delta : l - delta + len(temp_block)]) / sum(temp_block[-delta:]) > (k + 0.5) / move_node:
                            if len(temp_nodelist[l - delta - 1]) < temp_block[l - delta - 1]:
                                temp_nodelist[l - delta - 1] += [temp_nodelist[j].pop()]
                                found = True
                                break
                # No position. Miss at the first place. Return and add back to the first block.
                if not found:
                    temp_nodelist[-delta] += [temp_nodelist[j].pop()]

        if sum(temp_block[-delta:]) > 1:
            temp_prob[-delta:, -delta:] = (degree - sum(temp_prob[-1] * temp_block)) / (sum(temp_block[-delta:]) - 1.)
        # Random prob is also okay.
        else:
            temp_prob[-delta:, -delta:] = 0.
    
        block_sizes.append(end_block_size)
        blocks.append(temp_block)
        probs.append(temp_prob)
        nodelists.append(temp_nodelist)
        if i != change_period:
            periods.append(pause_period)
        else:
            if reverse:
                periods.append(start_period)
            else:
                periods.append(end_period)

    if reverse:
        start_block_size, end_block_size = end_block_size, start_block_size
        block_sizes.reverse()
        blocks.reverse()
        probs.reverse()
        nodelists.reverse()

    graph_model = SBM_structure_random_graph(num_nodes, block_sizes[0], degree, blocks[0], probs[0], nodelists[0])

    num_period = len(blocks)

    parameters = zip(block_sizes, blocks, probs, nodelists)


########## DEFINE CHANGE POINTS ##########

total_time = sum(periods)

# to cumulated form
#change_points = np.cumsum(periods)[:-1]
change_points = np.array(periods[0:1]) + window_size // 2 - 1


########## UTILITY FUNCTIONS ##########

evaluate = auc_benefit_fan


########## EMBEDDING MODELS ##########

gae_model = GAE(num_nodes, epoch=epoch, hidden1=dim*2, hidden2=dim)
dw_model = DeepWalk(dimension=dim)


########## WINDOWS ##########

def gae_rwils(g):
    return embedding_random_walk(gae_model.embed(g), activate, pairwise_transition)

def dw_rwils(g):
    return embedding_random_walk(dw_model.embed(g), activate, pairwise_transition)

def rw(g):
    return random_walk(nx.adjacency_matrix(g))

window_gae = sliding_window(window_size, gae_rwils, estimate_method)
window_dw = sliding_window(window_size, dw_rwils, estimate_method)
window_org = sliding_window(window_size, rw, estimate_method)


########## AGGREGATE MODEL ##########

agg_model = aggregate_model_evaluation(a, b, individual=True)

agg_model.add_model(window_gae.insert_new_data)  # GAE:      1st idx=0
agg_model.add_model(window_dw.insert_new_data)   # DeepWalk: 1st idx=1
agg_model.add_evaluation(kl_score)               # KL:       2nd idx=0
agg_model.add_evaluation(smdl_score)             # SMDL:     2nd idx=1


########## TRAIN GRAPH MODELS ##########

g = graph_model.generate()
train_roc_score, test_roc_score = gae_model.train(g)


########## FILL WINDOWS ##########

ready = False
i = 0
period = window_size

while not ready:
    g = graph_model.generate()
    adj = nx.adjacency_matrix(g)

    ready_gae = not window_gae.fill(g)
    ready_dw = not window_dw.fill(g)
    ready_org = not window_org.fill(g)
    ready = ready_gae and ready_dw and ready_org
    i += 1

    print("Fill: {} / {}".format(i, period))


########## DYNAMIC SYNTHETIC DATA ##########

for parameter, period in zip(parameters, periods):
    # parameter update
    if change_type:
        deg, = parameter
        if deg != start_deg: # avoid first update from baseline
            graph_model.update(expected_deg=deg)
    else:
        block_size, block, prob, nodelist = parameter
        if block != start_block: # avoid first update from baseline
            graph_model = SBM_structure_random_graph(num_nodes, block_size, degree, block, prob, nodelist)

    t = 0
    for i in range(period):
        t += 1

        g = graph_model.generate()
        adj = nx.adjacency_matrix(g)

        ### Original ###                
        window_org.insert_new_data(g)
        window_org.calculate_scores()

        ### Aggregate ###
        agg_model.insert_new_data(g)

        graph_model.info()
        print("{} / {}".format(t, period))


########## SCORES ##########

aggs = agg_model.get_scores()
individual_scores = agg_model.get_individual_scores()

kl_gae = individual_scores[0][0]
smdl_gae = individual_scores[0][1]
kl_dw = individual_scores[1][0]
smdl_dw = individual_scores[1][1]

kl_org, smdl_org = window_org.get_scores()

aggs = normalize_score(aggs)

kl_gae = normalize_score(kl_gae)
kl_dw = normalize_score(kl_dw)
kl_org = normalize_score(kl_org)

smdl_gae = normalize_score(smdl_gae)
smdl_dw = normalize_score(smdl_dw)
smdl_org = normalize_score(smdl_org)


########## RESULTS ##########

print("RWiLS (AGG) AUC={}".format(evaluate(aggs, change_points, window_size)))
print("RWiLS: GAE (KL) AUC={}".format(evaluate(kl_gae, change_points, window_size)))
print("RWiLS: GAE (MDL) AUC={}".format(evaluate(smdl_gae, change_points, window_size)))
print("RWiLS: DeepWalk (KL) AUC={}".format(evaluate(kl_dw, change_points, window_size)))
print("RWiLS: DeepWalk (MDL) AUC={}".format(evaluate(smdl_dw, change_points, window_size)))
print("RW (KL) AUC={}".format(evaluate(kl_org, change_points, window_size)))
print("RW (MDL) AUC={}".format(evaluate(smdl_org, change_points, window_size)))
