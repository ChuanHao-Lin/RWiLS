import numpy as np
import networkx as nx
from random import random, randint, sample


class random_graph(object):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def graph_adj(self, graph):
        return nx.adjacency_matrix(graph)
        
    def generate(self):
        raise NotImplementedError
        
    def update(self, **kwargs):
        raise NotImplementedError
        
    def terminate(self):
        raise NotImplementedError
        
    def get_parameter(self):
        raise NotImplementedError

    def info(self):
        raise NotImplementedError

    def model_name(self):
        raise NotImplementedError


class naive_random_graph(random_graph):
    def __init__(self, num_nodes, edge_ratio):
        super(naive_random_graph, self).__init__(num_nodes)
        self.edge_ratio = edge_ratio
        
    def generate(self):
        return nx.fast_gnp_random_graph(self.num_nodes, self.edge_ratio)
        
    def update(self, edge_ratio=None, delta=None):
        assert edge_ratio or delta

        if edge_ratio:
            self.edge_ratio = edge_ratio
        else:
            self.edge_ratio += delta

        assert not self.terminate()

    def terminate(self):
        return (self.edge_ratio > 1) or (self.edge_ratio < 0)

    def get_parameter(self):
        return self.edge_ratio

    def info(self):
        print("Edge ratio = {}".format(self.edge_ratio))
        return self.edge_ratio

    def model_name(self):
        return "ER" # Erdos-Renyi


class stochastic_block_model(random_graph):
    def __init__(self, num_nodes, block_size, expected_deg, blocks=None, prob_matrix=None):
        super(stochastic_block_model, self).__init__(num_nodes)
        
        self.block_size = block_size
        self.exp_deg = expected_deg

        if blocks is None:
            self.blocks = self.init_blocks()
        else:
            self.blocks = blocks

        if prob_matrix is None:
            self.prob_matrix = self.init_prob()
        else:
            self.prob_matrix = prob_matrix

    def init_blocks(self):
        blocks = []
        rest_nodes = self.num_nodes
        for i in range(self.block_size):
            if i == self.block_size - 1:
                assert rest_nodes > 0
                blocks.append(rest_nodes)
            else:
                block_node = randint(1, rest_nodes - self.block_size + i + 1)
                blocks.append(block_node)
                rest_nodes -= block_node
        return blocks

    def init_prob(self):
        prob_matrix = np.zeros((self.block_size, self.block_size))
        try:
            init_ratio = self.exp_deg / (self.num_nodes - 1.)
        except:
            init_ratio = float(self.exp_deg) / self.num_nodes
        while (prob_matrix <= 0.).any() or (prob_matrix >= 1.).any():
            prob_matrix = np.zeros((self.block_size, self.block_size))
            for i in range(self.block_size):
                for j in range(i+1, self.block_size):
                    lower_ratio = random() * init_ratio
                    prob_matrix[i][j] = lower_ratio
                    prob_matrix[j][i] = lower_ratio
                if self.blocks[i] > 1:
                    prob_matrix[i][i] = (self.exp_deg - sum(prob_matrix[i] * self.blocks)) / (self.blocks[i] - 1.)
                # Single node block. Any value is fine
                else:
                    prob_matrix[i][i] = random()
        return prob_matrix

    def generate(self, nodelist=None):
        return nx.stochastic_block_model(self.blocks, self.prob_matrix, nodelist)
    
    def expect_degree(self):
        return np.dot(self.prob_matrix, self.blocks) - np.diag(self.prob_matrix)
        
    def update(self, **kwargs):
        raise NotImplementedError
        
    def terminate(self):
        return self.block_size <= 0 or self.exp_deg <= 0 or (self.prob_matrix < 0).any() or (self.prob_matrix > 1).any()
        
    def get_parameter(self):
        raise NotImplementedError

    def info(self):
        print("Nodes in blocks = {}".format(self.blocks))
        print("Expected degrees = {}".format(self.expect_degree()))
        return self.blocks, self.expect_degree()

    def model_name(self):
        return "SBM"


class SBM_structure_random_graph(stochastic_block_model):
    def __init__(self, num_nodes, block_size, expected_deg, blocks=None, prob_matrix=None, nodelist=None):
        super(SBM_structure_random_graph, self).__init__(num_nodes, block_size, expected_deg, blocks, prob_matrix)
        if nodelist is None:
            self.nodelist = list()
            start_node = 0
            for i in self.blocks:
                self.nodelist.append(list(range(start_node, start_node + i)))
                start_node += i
        else:
            self.nodelist = nodelist

    def generate(self):
        return super(SBM_structure_random_graph, self).generate(sum(self.nodelist, []))

    def update(self, block_size=None, delta=None):
        assert block_size or delta

        if block_size:
            delta = block_size - self.block_size
        
        # Update starts here
        if delta == 0:
            return

        # Merge blocks
        if delta < 0:
            for i in range(np.abs(delta)):
                temp = sample(range(self.block_size), 2)
                merge = temp[0]
                target = temp[1]

                # probability update
                for j in range(self.block_size):
                    new_prob = (self.blocks[merge] * self.prob_matrix[merge][j] + self.blocks[target] * self.prob_matrix[target][j]) / (self.blocks[merge] + self.blocks[target])
                    self.prob_matrix[merge][j] = new_prob
                    self.prob_matrix[j][merge] = new_prob                
                self.prob_matrix[merge][merge] = 0

                # node of blocks update
                self.blocks[merge] += self.blocks[target]
                self.nodelist[merge] += self.nodelist.pop(target)

                self.blocks[target] = 0

                self.prob_matrix[merge][merge] += (self.exp_deg - sum(self.prob_matrix[merge] * self.blocks)) / (self.blocks[merge] - 1)

                # remove the merged block
                self.blocks.pop(target)
                self.prob_matrix = np.delete(np.delete(self.prob_matrix, target, 0), target, 1)

                self.block_size -= 1

                assert not super(SBM_structure_random_graph, self).terminate()

        # Split blocks
        elif delta > 0:
            for i in range(delta):
                separate_point = randint(1, self.num_nodes)
                new_block = 0
                news = []
                for i in range(self.block_size):
                    if self.blocks[i] < separate_point or separate_point < 0:
                        new = int(self.blocks[i] ** 2. * random() / self.num_nodes / self.block_size)
                    else:
                        if self.blocks[i] != separate_point:
                            new = separate_point
                        else:
                            new = randint(min(1, self.blocks[i] - 1), self.blocks[i] - 1)
                    new_block += new
                    news.append(new)
                    separate_point -= self.blocks[i]
                
                # Safety check. Redo if error happens
                if new_block <= 0:
                    self.update(block_size, delta)
                    return
                    
                new_nodes = list()
                for i in range(self.block_size):
                    self.blocks[i] -= news[i]
                    new_nodes += self.nodelist[i][-news[i]:]
                    self.nodelist[i] = self.nodelist[i][:-news[i]]
                #self.blocks -= np.array(news)
                self.blocks.append(new_block)
                self.nodelist.append(new_nodes)
                news.append(0)
                self.prob_matrix = np.pad(self.prob_matrix, (0, 1), mode="constant")
                for i in range(self.block_size):
                    self.prob_matrix[self.block_size][i] = sum(self.prob_matrix[i] * news) / new_block
                    self.prob_matrix[i][self.block_size] = self.prob_matrix[self.block_size][i]
                if self.blocks[self.block_size] > 1:
                    self.prob_matrix[self.block_size][self.block_size] = (self.exp_deg - sum(self.prob_matrix[self.block_size] * self.blocks)) / (self.blocks[self.block_size] - 1)
                else:
                    self.prob_matrix[self.block_size][self.block_size] = random()
                
                self.block_size += 1

    def get_parameter(self):
        return self.block_size

    def info(self):
        return super(SBM_structure_random_graph, self).info()


class SBM_parameter_random_graph(stochastic_block_model):
    def __init__(self, num_nodes, block_size, expected_deg, blocks=None, prob_matrix=None):
        super(SBM_parameter_random_graph, self).__init__(num_nodes, block_size, expected_deg, blocks, prob_matrix)

    def generate(self):
        return super(SBM_parameter_random_graph, self).generate()

    def update(self, expected_deg=None, delta=None):
        assert expected_deg or delta

        if expected_deg:
            self.exp_deg = expected_deg
        else:
            self.exp_deg += delta

        assert not super(SBM_parameter_random_graph, self).terminate()

        self.prob_matrix = super(SBM_parameter_random_graph, self).init_prob()

    def get_parameter(self):
        return self.exp_deg

    def info(self):
        return super(SBM_parameter_random_graph, self).info()


### Not finished yet ###
class gaussian_random_graph(random_graph):
    def __init__(self, num_nodes, cluster_mean, cluster_var, prob_in, prob_out):
        super(gaussian_random_graph, self).__init__(num_nodes)
        self.v = cluster_mean / cluster_var
        self.cluster_mean = cluster_mean
        self.prob_in = prob_in
        self.prob_out = prob_out
        
    def generate(self):
        return nx.gaussian_random_partition_graph(self.num_nodes, self.cluster_mean, self.v, self.prob_in, self.prob_out)
        
    def update(self, delta=0.1):
        pass
        
    def terminate(self):
        pass

    def get_parameter(self):
        pass

    def info(self):
        return "Cluster Mean = {0}; Cluster Var = {1}".format(self.cluster_mean, self.cluster_mean / self.v)

    def model_name(self):
        return "Gaussian"