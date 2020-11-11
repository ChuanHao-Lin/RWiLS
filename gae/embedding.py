### This file is meant to be a wrapper, which allows users to use GAE without explictly calling tensorflow
### Most of the contents are modified from train.py

from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU)
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .optimizer import OptimizerAE, OptimizerVAE
from .model import GCNModelAE, GCNModelVAE
from .preprocessing import *


# define tensorflow variables that might be accessed in other scope
sess = tf.Session()
flags = tf.app.flags
FLAGS = flags.FLAGS


class GAE(object):
    def __init__(self, num_node, epoch=200, model="gcn_ae", use_feature=False, learning_rate=0.01, 
                 hidden1=32, hidden2=16, weight_decay=0., dropout=0., features=None):
        self.num_node = num_node

        self.placeholders = { 'features': tf.sparse_placeholder(tf.float32),
                              'adj': tf.sparse_placeholder(tf.float32),
                              'adj_label': tf.sparse_placeholder(tf.float32),
                              'adj_norm': tf.sparse_placeholder(tf.float32),
                              'dropout': tf.placeholder_with_default(0., shape=()) }
        # default value
        flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', epoch, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
        flags.DEFINE_integer('hidden2', hidden2, 'Number of units in hidden layer 2.')
        flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
        flags.DEFINE_string('model', model, 'Model string.')
        flags.DEFINE_boolean('features', use_feature, 'Whether to use features')

        # Start building model
        self.model = self.create_model(num_node, features)
        self.opt = self.create_optimizer()
        self.initialize_session()

    # Initialize manually so that the it won't conflict with the optimizer
    def initialize_session(self):
        sess.run(tf.global_variables_initializer())

    def define_variables(self, epoch=200, model="gcn_ae", use_feature=False, 
                        learning_rate=0., hidden1=32, hidden2=16, weight_decay=0., dropout=0.):
        self.delete_variables()
        
        # Settings
        flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
        flags.DEFINE_integer('epochs', epoch, 'Number of epochs to train.')
        flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
        flags.DEFINE_integer('hidden2', hidden2, 'Number of units in hidden layer 2.')
        flags.DEFINE_float('weight_decay', weight_decay, 'Weight for L2 loss on embedding matrix.')
        flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')

        flags.DEFINE_string('model', model, 'Model string.')
        flags.DEFINE_boolean('features', use_feature, 'Whether to use features')

    def delete_variables(self):
        delattr(FLAGS, "learning_rate")
        delattr(FLAGS, "epochs")
        delattr(FLAGS, "hidden1")
        delattr(FLAGS, "hidden2")
        delattr(FLAGS, "weight_decay")
        delattr(FLAGS, "dropout")
        delattr(FLAGS, "model")
        delattr(FLAGS, "features")

    def create_model(self, num_nodes, features=None):
        if features is None:
            features = dummy_features(num_nodes)
        features = sparse_to_tuple(features.tocoo())

        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        if FLAGS.model == 'gcn_ae':
            return GCNModelAE(self.placeholders, num_features, features_nonzero)
        elif FLAGS.model == 'gcn_vae':
            return GCNModelVAE(self.placeholders, num_features, num_nodes, features_nonzero)
        
        return None

    def create_optimizer(self):
        with tf.name_scope('optimizer'):
            if FLAGS.model == 'gcn_ae':
                return OptimizerAE(preds=self.model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_label'],
                                                                               validate_indices=False), [-1]),
                                   adj=self.placeholders['adj'])
            elif FLAGS.model == 'gcn_vae':
                return OptimizerVAE(preds=self.model.reconstructions,
                                    labels=tf.reshape(tf.sparse_tensor_to_dense(self.placeholders['adj_label'],
                                                                                validate_indices=False), [-1]),
                                    model=self.model, 
                                    adj=self.placeholders['adj'])

        return None

    def train(self, G, val_ratio=0.2, test_ratio=0.1, features=None):
        assert val_ratio >= 0 and test_ratio >= 0 and val_ratio + test_ratio <= 1 

        adj = nx.adjacency_matrix(G)
        adj_train, val_edges, val_edges_false, test_edges, test_edges_false = split_graph(adj, val_ratio, test_ratio)

        if features is None:
            features = dummy_features(adj_train.shape[0])
        features = sparse_to_tuple(features.tocoo())

        val_roc_score = []

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = preprocess_graph(adj_train)
        # fit other dimension for feed_dict
        adj_train = sparse_to_tuple(adj_train)

        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_train, adj_label, adj_norm, features, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([self.opt.opt_op, self.opt.cost, self.opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            roc_curr, ap_curr = self.get_roc_score(feed_dict, adj, val_edges, val_edges_false)
            val_roc_score.append(roc_curr)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        if test_ratio:
            test_roc_score = self.test(adj, test_edges, test_edges_false)
            return val_roc_score, test_roc_score
        else:
            return val_roc_score

    # Modified from train. It won't validate nor test. Faster
    def fit(self, G, features=None):
        adj = nx.adjacency_matrix(G)

        if features is None:
            features = dummy_features(self.num_node)
        features = sparse_to_tuple(features.tocoo())

        adj_label = adj + sp.eye(self.num_node)
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = preprocess_graph(adj)
        # fit other dimension for feed_dict
        adj = sparse_to_tuple(adj)

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj, adj_label, adj_norm, features, self.placeholders)
            feed_dict.update({self.placeholders['dropout']: FLAGS.dropout})

            # Run single weight update
            outs = sess.run([self.opt.opt_op, self.opt.cost, self.opt.accuracy], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            avg_accuracy = outs[2]

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy), "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

    def test(self, adj, test_edges, test_edges_false, features=None):
        if features is None:
            features = dummy_features(self.num_node)
        features = sparse_to_tuple(features.tocoo())

        adj_label = adj + sp.eye(self.num_node)
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = preprocess_graph(adj)
        # fit other dimension for feed_dict
        adj_test = sparse_to_tuple(adj)

        feed_dict = construct_feed_dict(adj_test, adj_label, adj_norm, features, self.placeholders)

        roc_score, ap_score = self.get_roc_score(feed_dict, adj, test_edges, test_edges_false)
        print('Test ROC score: ' + str(roc_score))
        print('Test AP score: ' + str(ap_score))

        return roc_score, ap_score

    def embed(self, G, features=None):
        adj = nx.adjacency_matrix(G)

        if not features:
            features = dummy_features(self.num_node)
        features = sparse_to_tuple(features.tocoo())

        adj_label = adj + sp.eye(self.num_node)
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = preprocess_graph(adj)
        # fit other dimension for feed_dict
        adj = sparse_to_tuple(adj)

        feed_dict = construct_feed_dict(adj, adj_label, adj_norm, features, self.placeholders)
        feed_dict.update({self.placeholders['dropout']: 0})

        emb = sess.run(self.model.z_mean, feed_dict=feed_dict)
        return emb

    def get_roc_score(self, feed_dict, adj, edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({self.placeholders['dropout']: 0})
            emb = sess.run(self.model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_rec = np.zeros(adj.shape)
        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)

        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def save_model(self, path):
        tf.train.Saver(sess, path)

    def load_model(self, path):
        tf.train.Saver().restore(sess, path)
