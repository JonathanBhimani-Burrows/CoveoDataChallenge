import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.data_processing import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances



def _node_embedding(data, city_totals_l):
    '''
    a function that creates the node embedding
    :param data: raw data
    :param city_count_l: counts of cities per country
    :return: node embedding
    '''

    cities, _ = list(zip(*city_totals_l))
    #note: this method of making a nested dict is ugly, but the comment right below gave very weird results
    #node_embedding = dict.fromkeys(_lookup, dict.fromkeys(cities, 0))
    node_embedding = dict()
    for it in cities:
        node_embedding[it] = {}
        for j in cities:
            node_embedding[it][j] = 0

    for item in data['cities'].iteritems():
        city_list = item[1][0].split(', ')
        if len(city_list) > 1:
            for _city in city_list:
                for __city in city_list:
                    if _city != __city:
                        node_embedding[_city][__city] += 1
    return node_embedding

def _node_distance(node_embedding):
    '''
    a function that generates the node distances from the node embeddings
    :param node_embedding: dict of node embeddings
    :return: matrix of distances
    '''
    node_embedding_l = []
    for item in node_embedding:
        node_embedding_l.append(list(node_embedding[item].values()))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(node_embedding_l)
    #plot_features(pca_result)
    node_distance = np.zeros((pca_result.shape[0],pca_result.shape[0]))
    for i in range(len(node_distance)):
        for j in range(len(node_distance)):
            node_distance[i][j] = euclidean_distances(pca_result[i].reshape(1,-1),pca_result[j].reshape(1,-1))
    #normalize the node distances
    node_distance[node_distance==0] = 0.0000001
    node_distance = (node_distance - node_distance.min(axis=1))/(node_distance.max(axis=1) - node_distance.min(axis=1))
    node_distance[node_distance == 0] = 10000
    return node_distance

def calc_score(node_distance, current, previous, prediction):
    pass



def q_4(data, city_count_l, city_totals_l,country_list, predictions_V1, predictions_V2, predictions_V3,
                        predictions_Q3):
    '''
    a function that calculates relative metrics for each approach
    :param data:
    :param city_count_l:
    :param city_totals_l:
    :param country_list:
    :return: 0
    '''
    node_embedding = _node_embedding(data, city_totals_l)
    node_distance = _node_distance(node_embedding)




