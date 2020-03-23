import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.data_processing import *
import random
import copy

def _track_predictions(counts, idx):
    '''
    a function that tracks the cities visited if the probabilities of a row go to 0, so an independent
    high probabily option can be chosen
    :param counts: counts of each city
    :param idx: chosen city index
    :return: counts with city index set to 0
    '''
    counts[idx] = 0
    return counts


def _mask_col(mat, col):
    '''
    a function that masks the columns that have already been visited
    :param col: column to be masked
    :return: hop matrix with column masked
    '''
    if mat.ndim == 2:
        mat[:, col] = 0
    elif mat.ndim == 3:
        mat[:, :, col] = 0
    else:
        raise ValueError('Wrong matrix dimension')
    return mat


def _mask_previous(hops, previous, counts, cities, city_lookup):
    if len(previous) != 0:
        for item in previous:
            hops = _mask_col(hops, int(city_lookup[item]))
            counts = _track_predictions(counts, int(city_lookup[item]))
        starting_city = previous[-1]
    else:
        starting_city = cities[np.argmax(counts)]
        counts = _track_predictions(counts, np.argmax(counts))
    return hops, counts, starting_city

def _choose_prediction(hops, options, counts, inverse_city_lookup,predictions):
    choice = random.choices(np.arange(hops.shape[1]), weights=options)
    print("The next suggested city is", inverse_city_lookup[choice[0]])
    predictions.append(inverse_city_lookup[choice[0]])
    hops = _mask_col(hops, choice[0])
    counts = _track_predictions(counts, choice[0])
    return hops, options, counts, choice, predictions

def V1(city_totals_l, previous, n, city_lookup):
    '''
    a function that prints suggested city (or cities) based on an independence assumption
    :param city_totals_l: total times a city has been visited
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :return: 0
    '''
    predictions = []
    if len(previous) > 0:
        for item in previous:
            city_totals_l.pop(int(city_lookup[item]))
    for i in range(n):
        print("The next suggested city is",city_totals_l[i][0])
        predictions.append(city_totals_l[i][0])
    return predictions




def V2(city_totals_l,k_hops_2p2, previous, n, city_lookup, inverse_city_lookup):
    '''
    a function that prints suggested city (or cities) based on a markov assumption
    :param
    :param k_hops_2p2: 1hop matrix of potential locations to visit
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :param inverse_city_lookup: inverse lookup table for city indices
    :return: 0
    '''
    predictions = []
    cities, counts = list(zip(*city_totals_l))
    counts = np.asarray(counts)
    #mask cities already searched
    k_hops_2p2, counts, starting_city = _mask_previous(k_hops_2p2, previous, counts, cities, city_lookup)
    #make suggestions
    options = k_hops_2p2[int(city_lookup[starting_city]), :]
    total_cities = len(previous) + n
    c = 0
    for i in range(n):
        if options.sum() != 0:
            k_hops_2p2, options, counts, choice, predictions = _choose_prediction(k_hops_2p2, options, counts,
                                                                                  inverse_city_lookup, predictions)
            options = k_hops_2p2[choice[0], :]

        else:
            # if the row probabilities are all 0, choice the highest probability independent choice then continue
            if counts.sum() != 0:
                choice = np.argmax(counts)
                print("The next suggested city is", cities[choice])
                predictions.append(cities[choice])
                counts = _track_predictions(counts, choice)
                options = k_hops_2p2[choice, :]
            else:
                #if all the independent probabilities are used, start from beginning and loop through
                print("The next suggested city is", cities[c])
                predictions.append(cities[c])
                c += 1
    return predictions



def V3(city_totals_l,k_hops_2p3, previous, n, city_lookup, inverse_city_lookup):
    '''
    a function that prints suggested city (or cities) based n-previous visited cities
    :param k_hops_2p3: khop matrix of potential locations to visit
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :param inverse_city_lookup: inverse lookup table for city indices
    :return: 0
    '''
    def _run_down_hops(k_hops_2p3, counts,cities, n, k, inverse_city_lookup, predictions):
        '''
        helper loop function that does the iterative looping down the hop values
        :param k_hops_2p3: khop matrix of potential locations to visit
        :param counts: number of counts per city
        :param n: number of recommendations requested
        :return: updated khop matrix of potential locations to visit, updated number of counts per city
        '''

        options = k_hops_2p3[k - 2, int(city_lookup[starting_city]), :]
        flag = False
        for i in range(n):
            # check loop indices
            for j in range(k - 2, -1, -1):
                if flag:
                    options = k_hops_2p3[j, choice[0], :]
                flag = True
                if options.sum() != 0:
                    k_hops_2p3, options, counts, choice, predictions = _choose_prediction(k_hops_2p3, options, counts,
                                                                                          inverse_city_lookup,predictions)

                    break

                print('looping {} for {} prediction'.format(j, i))
                if j == 0:
                    choice = [0]
                    print("The next suggested city is", inverse_city_lookup[choice[0]])
                    predictions.append(inverse_city_lookup[choice[0]])
                    counts = _track_predictions(counts, choice[0])
                    k_hops_2p3 = _mask_col(k_hops_2p3, choice[0])
            flag = False
            options = k_hops_2p3[k - 2, choice[0], :]
        return k_hops_2p3, counts, inverse_city_lookup[choice[0]]

    def _generate_subdivisions(a, b):
        '''
        helper function that generates subdivisions for longer search strings
        :param a: initial length
        :param b: increment width
        :return: divided searches
        '''
        div = []
        while(a-b >= 0):
            div.append(b)
            a -= b
            if a < b:div.append(a)
        return div


    predictions = []
    cities, counts = list(zip(*city_totals_l))
    counts = np.asarray(counts)
    #mask cities already searched
    k_hops_2p3, counts, starting_city = _mask_previous(k_hops_2p3, previous, counts, cities, city_lookup)
    length = len(previous) + n

    hops = k_hops_2p3.shape[0] + 1

    if length <= hops:
        #eg prev = 4, n = 3 -> length = 7
        # use k = 7 starting at starting city
        k = length
        _, _, predictions = _run_down_hops(k_hops_2p3, counts,cities, n, k,inverse_city_lookup, predictions)


    elif (len(previous) < hops and length >= hops ):
        # eg prev = 9, n = 5 -> length = 14
        # eg prev = 9, n = 15 -> length = 24 ewwwww
        # use k = 11 up until 11, then finish delta with remaining k
        k = hops
        # subdivide the total number into k increments
        div = _generate_subdivisions(length, k)
        for item in div:
            k_hops_2p3, counts, predictions = _run_down_hops(k_hops_2p3, counts,cities, int(item), k,
                                                             inverse_city_lookup, predictions)


    elif (len(previous) > hops and n > hops):
        #eg prev = 13, n = 14 -> start
        #use k = 11 up until 11, then finish delta with remaining k
        #recall
        k = hops

        #subdivide the total number into k increments
        div = _generate_subdivisions(n, k)

        for item in div:
            k_hops_2p3, counts, predictions = _run_down_hops(k_hops_2p3, counts,cities, int(item), k,
                                                             inverse_city_lookup, predictions)

    elif (len(previous) > hops and n < hops):
        #eg prev = 13, n = 5 -> start
        #use k = 5 starting at starting city
        k = n
        _, _, predictions = _run_down_hops(k_hops_2p3, counts,cities, n, k, inverse_city_lookup, predictions)

    else:
        print("Corner case detected")
    return predictions






def q_2(data, city_count_1, city_totals_l,country_list, previous, to_search):


    print("++++++++++++++Question 2+++++++++++++++++++")
    city_lookup = dict()
    inverse_city_lookup = dict()
    cities, counts = list(zip(*city_totals_l))
    c = 0
    for item in cities:
        city_lookup[item] = c
        inverse_city_lookup[c] = item
        c += 1
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX']
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX',"Stockton CA", 'San Jose CA', 'Seattle WA', 'Portland OR', 'Long Beach CA']
    #previous = []
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX',"Stockton CA", 'San Jose CA', 'Seattle WA', 'Portland OR', 'Long Beach CA', 'Los Angeles CA', 'Toronto ON', 'Madison WI']
    #to_search = 10
    hop_depth = 10
    print("++++++++++++++Question 2.1+++++++++++++++++++")
    #make deepcopy of citytotals so allow for popping
    city_totals_l_copy = copy.deepcopy(city_totals_l)
    predictions_V1 = V1(city_totals_l_copy, previous, to_search, city_lookup)
    print("++++++++++++++Question 2.2+++++++++++++++++++")
    k_hops_2p2 = create_hops(data, cities, city_lookup)
    k_hops_2p3 = create_hops(data, cities, city_lookup, hop_depth=hop_depth)
    predictions_V2 = V2(city_totals_l, k_hops_2p2, previous, to_search, city_lookup, inverse_city_lookup)
    print("++++++++++++++Question 2.3+++++++++++++++++++")
    predictions_V3 = V3(city_totals_l, k_hops_2p3, previous, to_search, city_lookup, inverse_city_lookup)
    return predictions_V1, predictions_V2, predictions_V3








