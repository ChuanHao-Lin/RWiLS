import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.special import loggamma, digamma, polygamma
from scipy.sparse.linalg import eigs
from sklearn.metrics import auc
from random import random
import sys
from dirichlet import dirichlet

### NOTE: Difference between loggamma() and gammaln() ###
#   loggamma(negative) = Nan
#   gammaln(negative) = inf
#   loggamma(complex) = complex
#   gammaln(complex) = undefined
### Dirichlet.py chooses gammaln(). Need real vectors ###


err = 1e-7


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLu(x):
    return x * (x > 0)


# Stationary vector for Markov Chain
def fast_markov_vector(M):
    vec = np.real(eigs(M.astype(float), 1)[1].reshape(M.shape[0]))
    return np.real(vec * np.conj(vec))
    

def markov_vector(M, non_zero=True):
    values, vectors = np.linalg.eig(M)

    idx = np.argmin(abs(values - 1))
    vec = np.real(vectors[:, idx] * np.conj(vectors[:, idx]))
            
    # Pad 0 with a small number (for MLE)
    if non_zero:
        # min value
        minval = np.min(vec[np.nonzero(vec)])
        minval = min(minval / 10 + (minval == 0), minval, err / 10)
        vec[np.nonzero(vec == 0)] = minval
                
    return vec


def norm_transition(transition):        
    # Remove diagonal elements
    np.fill_diagonal(transition, 0.)

    normalization = transition.sum(axis=0, keepdims=False)
    # Check 0 in case the term stay in diagonal
    for i in np.nonzero(normalization == 0)[0]:
        normalization[i] = 1.
        try:
            transition[:, i] = 1. / (transition.shape[0] - 1)
            transition[i, i] = 0.
        # Divide by 0. 1 point only. Theoretically impossible
        except:
            transition[:, i] = 1.

    return transition / normalization


def inner_transition(emb, active_function=lambda x: x):
    transition = active_function(np.dot(emb, emb.T))
    
    return norm_transition(transition)


def kl_dirichlet(alpha, beta):
    alpha_0 = sum(alpha)
    beta_0 = sum(beta)
    diff = alpha - beta
    return loggamma(alpha_0) - loggamma(beta_0) \
          - sum(loggamma(alpha)) + sum(loggamma(beta)) \
          + sum(diff * digamma(alpha)) \
          - digamma(alpha_0) * sum(diff)


def reverse_kl_dirichlet(alpha, beta):
    return kl_dirichlet(beta, alpha)


def jeffreys_dirichlet(alpha, beta):
    alpha_0 = sum(alpha)
    beta_0 = sum(beta)
    diff = alpha - beta
    return sum(diff * (digamma(alpha) - digamma(beta))) \
          - (digamma(alpha_0) - digamma(beta_0)) * sum(diff)


def nml_code_length(alpha, window_size):
    try:
        alpha_0 = sum(alpha)
        dim = len(alpha)
    except:
        alpha_0 = alpha
        dim = 1
    return window_size * (sum(loggamma(alpha)) - loggamma(alpha_0) - sum((alpha - 1) * (digamma(alpha) - digamma(alpha_0)))) + log_nml_c_dirichlet(dim, window_size)


def embedding_random_walk(emb, active, pairwise_transition):
    transition = pairwise_transition(emb, active)
    vec = fast_markov_vector(transition)

    return vec


def random_walk(adj):
    transition_org = norm_transition(adj.toarray())
    vec_org = fast_markov_vector(transition_org)

    return vec_org


def normalize_score(scores):
    scores = np.array(scores)
    scores -= np.min(scores)
    scores /= np.max(scores)
        
    return scores


def MDL_threshold(window_size, dim, significant=0.05):
    return dim / 2 * np.log(window_size) - np.log(significant)


# Evaluation method. Implemented according to SMDL paper
def auc_benefit_fan(scores, change_points, tolerate=100, threshold_step=0.0001):
    change_points = np.array(change_points)

    threshold = 0.

    benefits = []
    false_alarms = []

    while threshold <= 1.:
        benefit = 0.
        false_alarm = 0.
        for time in range(len(scores)):
            if scores[time] > threshold:
                predict = np.min(np.abs(change_points - time))
                if predict < tolerate:
                    benefit += 1 - predict / tolerate
                else:
                    false_alarm += 1

        benefits.append(benefit)
        false_alarms.append(false_alarm)

        threshold += threshold_step

    if np.max(benefits) != 0:
        benefits /= np.max(benefits)
    if np.max(false_alarms) != 0:
        false_alarms /= np.max(false_alarms)
    
    return auc(false_alarms, benefits)


# Evaluation method. Implemented according to TREE paper (Graph Partitioning)
def auc_benefit_far(scores, change_points, tolerate=3, threshold_step=0.0001):
    change_points = np.array(change_points)

    threshold = 0.

    benefits = []
    false_alarms = []

    while threshold <= 1.:
        benefit = 0.
        false_alarm = 0.
        total_alarm = 0.
        for time in range(len(scores)):
            if scores[time] > threshold:
                total_alarm += 1
                predict = np.min(np.abs(change_points - time))
                if predict < tolerate:
                    benefit += 1 - predict / tolerate
                else:
                    false_alarm += 1

        if total_alarm != 0:
            false_alarm /= total_alarm

        benefits.append(benefit)
        false_alarms.append(false_alarm)

        threshold += threshold_step

    false_alarms, benefits = map(list, zip(*sorted(set(zip(false_alarms, benefits)))))
    
    # AUC
    return auc(false_alarms, benefits)


def SNML(adj, tran):
        return np.sum(np.where(np.logical_and(tran > 0, tran < 1), -np.log(adj * tran + (1 - adj) * (1 - tran)), 0))


def SNML_ground_truth(v_node, p_mat):
    v_node = np.array(v_node)
    p_mat = np.array(p_mat)
    code = -p_mat * np.log(p_mat) - (1 - p_mat) * np.log(1 - p_mat)
    node_combination = np.dot(np.reshape(v_node, (-1, 1)), np.reshape(v_node, (1,-1))) - np.diag(v_node)
    return 0.5 * np.sum(code * node_combination)
    

def MC_integrate(func, n_range, trials):
    s = 0
    
    def sample():
        return [(i[1] - i[0]) * random() + i[0] for i in n_range]
    
    for _ in range(trials):
        s += func(sample())
    s /= trials
    for a, b in n_range:
        s *= b - a
    return s


def trigamma(x):
    return polygamma(1, x)


def log_fisher(alpha):
    alpha_0 = np.sum(alpha)
    return np.log(trigamma(alpha_0)) + np.sum(np.log(trigamma(alpha))) + np.log((1 / trigamma(alpha_0) + sum(1 / trigamma(alpha))))


# The method uses geometry mean instead of arithmetic mean as Monte Carlo integral does (as above)
# This is to avoid numerical hazards
def log_root_I_integration(N, trials=1000):
    a = 0
    b = sys.maxsize

    s = 0
    for _ in range(trials):
        # sample uniformly from [a, b]
        s += log_fisher([(b - a) * random() + a for _ in range(N)])
    s /= trials
    s += np.log(b - a) * N

    return s / 2


# Dynamic Programming
log_c = dict()
def log_nml_c_dirichlet(dim, window_size):
    if dim not in log_c:
        log_c[dim] = dict()
    if window_size not in log_c[dim]:
        log_c[dim][window_size] = dim / 2 * np.log(window_size / 2 / np.pi) + np.log(log_root_I_integration(dim))
    
    return log_c[dim][window_size]


def dirichlet_estimation_all(window, estimate_method="fixedpoint"):
    return dirichlet.mle(window, method=estimate_method)


def dirichlet_estimation_half(window, estimate_method="fixedpoint"):
    window_size = len(window)

    before = dirichlet.mle(window[:window_size//2], method=estimate_method)
    after = dirichlet.mle(window[window_size//2:], method=estimate_method)
    return before, after


def kl_score(window, estimate_method="fixedpoint"):
    before, after = dirichlet_estimation_half(window, estimate_method)
    return kl_dirichlet(after, before)


def smdl_score(window, estimate_method="fixedpoint"):
    window_size = len(window)

    before, after = dirichlet_estimation_half(window, estimate_method)
    alpha = dirichlet_estimation_all(window, estimate_method)
    return nml_code_length(alpha, window_size) - nml_code_length(before, window_size//2) - nml_code_length(after, window_size - window_size//2)
