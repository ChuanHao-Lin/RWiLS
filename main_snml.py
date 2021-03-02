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
from utility.sliding import sliding_window
# dirichlet
from dirichlet import dirichlet

# embed
from gae.embedding import GAE
from DeepWalk.embedding import DeepWalk


########## PARAMETER SETTING ##########

window_size = 10
dim = 32

epoch = 200

a = 100 # aggregation lower bound
b = 1000 # aggregation upper bound

pairwise_transition = inner_transition # distance_transition inner_transition
estimate_method = "fixedpoint" # meanprecision fixedpoint
activate = ReLu


########## RANDOM GRAPH PARAMETERS ##########

num_nodes = 1000
block_size = 5
degree = 10.


########## PROBLEM SETTING ##########

# True for parameter change. False for structure change
change_type = True

start_period = 20
end_period = 20

# gradual change (for both parameter change and structure change)
change_period = 1
pause_period = 1

# parameter change
start_deg = 10.
end_deg = 10.1

# structure change
start_block_size = 4
end_block_size = 5


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


########## GAUSSION FUNCTIONS ##########

def kl_gaussian(window):
    before_mean = np.mean(window[:window_size//2])
    before_var = np.var(window[:window_size//2])
    after_mean = np.mean(window[window_size//2:])
    after_var = np.var(window[window_size//2:])
    return (np.log(before_var / after_var) + (after_var + (after_mean - before_mean) ** 2) / before_var - 1) / 2

def smdl_gaussian(window):
    before_std = np.std(window[:window_size//2])
    after_std = np.std(window[window_size//2:])
    all_var = np.var(window)
    return window_size / 2 * np.log(all_var / before_std / after_std)


########## EMBEDDING MODELS ##########

gae_model = GAE(num_nodes, epoch=epoch, hidden1=dim*2, hidden2=dim)
dw_model = DeepWalk(dimension=dim)


########## WINDOWS ##########

def rwils(emb):
    return embedding_random_walk(emb, activate, pairwise_transition)

def rw(g):
    return random_walk(nx.adjacency_matrix(g))

def snml_codelength(g, model):
    adj = nx.adjacency_matrix(g)
    emb = pairwise_transition(model.embed(g), activate)
    return SNML(adj.toarray(), emb)

def embed(g, model):
    return pairwise_transition(model.embed(g), activate)

window_gae = sliding_window(window_size, [kl_score, smdl_score])
window_dw = sliding_window(window_size, [kl_score, smdl_score])
window_org = sliding_window(window_size, [kl_score, smdl_score])

window_snml_gae = sliding_window(window_size, [kl_gaussian, smdl_gaussian])
window_snml_dw = sliding_window(window_size, [kl_gaussian, smdl_gaussian])


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

    emb_gae = embed(g, gae_model)
    emb_dw = embed(g, dw_model)

    vec_gae = rwils(emb_gae)
    vec_dw = rwils(emb_dw)
    vec_org = rw(g)

    snml_gae = [SNML(adj.toarray(), emb_gae)]
    snml_dw = [SNML(adj.toarray(), emb_dw)]

    ready_gae = not window_gae.fill(vec_gae)
    ready_dw = not window_dw.fill(vec_dw)
    ready_org = not window_org.fill(vec_org)
    ready_snml_gae = not window_snml_gae.fill(snml_gae)
    ready_snml_dw = not window_snml_dw.fill(snml_dw)
    ready = ready_gae and ready_dw and ready_org and ready_snml_gae and ready_snml_dw
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

        emb_gae = embed(g, gae_model)
        emb_dw = embed(g, dw_model)

        vec_gae = rwils(emb_gae)
        vec_dw = rwils(emb_dw)
        vec_org = rw(g)

        snml_gae = [SNML(adj.toarray(), emb_gae)]
        snml_dw = [SNML(adj.toarray(), emb_dw)]

        window_gae.insert_new_data(vec_gae)
        window_dw.insert_new_data(vec_dw)
        window_org.insert_new_data(vec_org)

        window_snml_gae.insert_new_data(snml_gae)
        window_snml_dw.insert_new_data(snml_dw)

        graph_model.info()
        print("{} / {}".format(t, period))


########## SCORES ##########

kl_gae, smdl_gae = window_gae.get_scores()
kl_dw, smdl_dw = window_dw.get_scores()
kl_org, smdl_org = window_org.get_scores()
kl_snml_gae, smdl_snml_gae = window_snml_gae.get_scores()
kl_snml_dw, smdl_snml_dw = window_snml_dw.get_scores()

kl_gae = normalize_score(kl_gae)
kl_dw = normalize_score(kl_dw)
kl_org = normalize_score(kl_org)
kl_snml_gae = normalize_score(kl_snml_gae)
kl_snml_dw = normalize_score(kl_snml_dw)

smdl_gae = normalize_score(smdl_gae)
smdl_dw = normalize_score(smdl_dw)
smdl_org = normalize_score(smdl_org)
smdl_snml_gae = normalize_score(smdl_snml_gae)
smdl_snml_dw = normalize_score(smdl_snml_dw)


########## RESULTS ##########

print("SNML => RWiLS: GAE (KL) AUC={}".format(evaluate(kl_snml_gae, change_points, window_size // 2)))
print("SNML => RWiLS: GAE (MDL) AUC={}".format(evaluate(smdl_snml_gae, change_points, window_size // 2)))
print("SNML => RWiLS: DeepWalk (KL) AUC={}".format(evaluate(kl_snml_dw, change_points, window_size // 2)))
print("SNML => RWiLS: DeepWalk (MDL) AUC={}".format(evaluate(smdl_snml_dw, change_points, window_size // 2)))

print("RWiLS: GAE (KL) AUC={}".format(evaluate(kl_gae, change_points, window_size // 2)))
print("RWiLS: GAE (MDL) AUC={}".format(evaluate(smdl_gae, change_points, window_size // 2)))
print("RWiLS: DeepWalk (KL) AUC={}".format(evaluate(kl_dw, change_points, window_size // 2)))
print("RWiLS: DeepWalk (MDL) AUC={}".format(evaluate(smdl_dw, change_points, window_size // 2)))
print("RW (KL) AUC={}".format(evaluate(kl_org, change_points, window_size // 2)))
print("RW (MDL) AUC={}".format(evaluate(smdl_org, change_points, window_size // 2)))
