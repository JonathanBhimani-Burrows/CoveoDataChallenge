import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from utils.matrix_processing import mask_previous
import copy


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
    #create embedding
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
        node_embedding_l.append(np.array(list(node_embedding[item].values())))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(node_embedding_l)
    #plot features, if desired (commented for now
    #plot_features(pca_result)

    node_distance = np.zeros((pca_result.shape[0],pca_result.shape[0]))
    for i in range(len(node_distance)):
        for j in range(len(node_distance)):
            node_distance[i][j] = euclidean_distances(node_embedding_l[i].reshape(1,-1),node_embedding_l[j].reshape(1,-1))
    #normalize rows
    node_distance[node_distance==0] = np.nan
    max = np.nanmax(node_distance, axis=1)
    min = np.nanmin(node_distance, axis=1)
    mat = (node_distance - min)/(max-min)
    node_distance = mat.T
    return node_distance


def calc_score(node_distance, prediction,city_lookup, GT, mode):
    '''
    a function that calculates the score of the predictions
    :param node_distance: node distance matrix
    :param prediction: algorithm predictions
    :param city_lookup: lookup table for cities
    :param prediction: ground truth labels
    :param GT: ground truth predictions
    :param mode: chooses classification or distance as a metric
    :return: scores
    '''
    vals = []
    for item in prediction:
        vals.append(city_lookup[item])
    results = []
    #pick between 2 different modes, if desired
    if mode == 'classification':
        for i in range(len(vals)):
            if prediction[i] == GT[i]:
                results.append(1)
            else:
                results.append(0)
    elif mode == 'distance':
        for i in range(len(vals)):
            if GT[i] == vals[i]:
                results.append(0)
            else:
                results.append(node_distance[GT[i]][vals[i]])
    results = sum(results)/len(results)
    return results


def q_4(data, city_totals_l, preds,previous, to_search, city_lookup, inverse_city_lookup):
    '''
    a function that calculates relative metrics for each approach
    :param data:
    :param city_count_l:
    :param city_totals_l:
    :param country_list:
    :return: 0
    '''
    #create embedding and distance matrices
    node_embedding = _node_embedding(data, city_totals_l)
    node_distance = _node_distance(node_embedding)
    GT = []
    GT_disp = []
    cities, counts = list(zip(*city_totals_l))
    cities = list(cities)
    counts = list(counts)
    #mask previously searched cities
    if len(previous) != 0:
        node_distance, counts, starting_city = mask_previous(node_distance, previous, counts, cities, city_lookup, np.nan)
        correct_idx = city_lookup[starting_city]
    else:
        correct_idx = 0
    #make a deepcopy so modifications to the node_distance aren't persisted
    node_distance_c = copy.deepcopy(node_distance)
    for i in range(to_search):
        correct = np.nanargmin(node_distance_c[correct_idx, :])
        GT.append(correct)
        GT_disp.append(inverse_city_lookup[correct])
        node_distance_c[:,correct] = 1000
        correct_idx = correct
    #calc scores
    print("++++++++++++++Question 4+++++++++++++++++++")
    print("Ground Truths: ", GT_disp)
    q = ['Question 2 V1','Question 2 V2','Question 2 V3','Question 3']

    #this part will be left in but should be ignored, as all the values usually come up as 0
    #c = 0
    #print("Classification Scores for each approach are: \n ")
    # for predictions in preds:
    #     score = calc_score(node_distance, predictions,city_lookup, GT, 'classification')
    #     print("{}: {} ".format(q[c],score))
    #     c += 1

    print("Distance Scores for each approach are: \n ")
    c = 0
    for predictions in preds:
        score = calc_score(node_distance, predictions,city_lookup, GT, 'distance')
        print("Predictions for {} are {}".format(q[c], predictions))
        print("{}: {} ".format(q[c],score))
        c += 1




