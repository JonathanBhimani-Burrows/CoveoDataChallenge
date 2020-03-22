import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.data_processing import *
import random


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


def V1(city_totals_1, previous, n, city_lookup):
    '''
    a function that prints suggested city (or cities) based on an independence assumption
    :param city_totals_1: total times a city has been visited
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :return: 0
    '''
    for item in previous:
        city_totals_1.pop(int(city_lookup[item]))
    for i in range(n):
        print("The next suggested city is",city_totals_1[i][0])




def V2(city_totals_1,k_hops_2p2, previous, n, city_lookup, inverse_city_lookup):
    '''
    a function that prints suggested city (or cities) based on a markov assumption
    :param k_hops_2p2: 1hop matrix of potential locations to visit
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :param inverse_city_lookup: inverse lookup table for city indices
    :return: 0
    '''
    cities, counts = list(zip(*city_totals_1))
    counts = np.asarray(counts)
    #mask cities already searched
    for item in previous:
        print(item)
        k_hops_2p2 = _mask_col(k_hops_2p2, int(city_lookup[item]))
        counts = _track_predictions(counts, int(city_lookup[item]))
    starting_city = previous[-1]
    #make suggestions
    options = k_hops_2p2[int(city_lookup[starting_city]), :]
    total_cities = len(previous) + n
    c = 0
    for i in range(n):
        if options.sum() != 0:
            choice = random.choices(np.arange(k_hops_2p2.shape[1]), weights=options)
            print("The next suggested city is", inverse_city_lookup[choice[0]])
            k_hops_2p2 = _mask_col(k_hops_2p2, choice[0])
            counts = _track_predictions(counts, choice[0])
            options = k_hops_2p2[choice[0], :]
        else:
            # if the row probabilities are all 0, choice the highest probability independent choice then continue
            if counts.sum() != 0:
                choice = np.argmax(counts)
                print("The next suggested city is", cities[choice])
                counts = _track_predictions(counts, choice)
                options = k_hops_2p2[choice, :]
            else:
                #if all the independent probabilities are used, start from beginning and loop through
                print("The next suggested city is", cities[c])
                c += 1



def V3(city_totals_1,k_hops_2p3, previous, n, city_lookup, inverse_city_lookup):
    '''
    a function that prints suggested city (or cities) based n-previous visited cities
    :param k_hops_2p3: khop matrix of potential locations to visit
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :param inverse_city_lookup: inverse lookup table for city indices
    :return: 0
    '''
    def _tri(k_hops_2p3, counts,n, k):
        '''
        internal loop function that does the iterative looping down the hop values
        :param k_hops_2p3: khop matrix of potential locations to visit
        :param counts: number of counts per city
        :param n: number of recommendations requested
        :return: updated khop matrix of potential locations to visit, updated number of counts per city
        '''
        options = k_hops_2p3[k - 2, int(city_lookup[starting_city]), :]
        for i in range(n):
            # check loop indices
            for j in range(k - 2, 0, -1):
                if options.sum() != 0:
                    print('options != 0')
                    print("sum is", options.sum())
                    print(k)
                    choice = random.choices(np.arange(k_hops_2p3.shape[1]), weights=options)
                    print("The next suggested city is", inverse_city_lookup[choice[0]])
                    k_hops_2p3 = _mask_col(k_hops_2p3, choice[0])
                    counts = _track_predictions(counts, choice[0])
                    break
                print('looping {} for {} prediction'.format(j, i))
                options = k_hops_2p3[j, choice[0], :]
            options = k_hops_2p3[k - 2, choice[0], :]
        return k_hops_2p3, counts


    cities, counts = list(zip(*city_totals_1))
    counts = np.asarray(counts)
    #mask cities already searched
    for item in previous:
        print(item)
        k_hops_2p3 = _mask_col(k_hops_2p3, int(city_lookup[item]))
        counts = _track_predictions(counts, int(city_lookup[item]))
    starting_city = previous[-1]
    length = len(previous) + n
    #check this +1 or not
    hops = k_hops_2p3.shape[0] + 1
    starting_city = previous[-1]
    if len(previous) == 0:
        for i in range(n):
            print("The next suggested city is", city_totals_1[i][0])
    elif length <= hops:
        #eg prev = 4, n = 3 -> length = 7
        # use k = 7 starting at starting city
        k = length
        _, _ = _tri(k_hops_2p3, counts,n, k)


    elif (len(previous) < hops and length >= hops ):
        # eg prev = 9, n = 5 -> length = 14
        # eg prev = 9, n = 15 -> length = 24 ewwwww
        # use k = 11 up until 11, then finish delta with remaining k
        div = []
        k = hops
        # subdivide the total number into k increments
        while (length - k >= 0):
            div.append(k)
            length -= k
            if length < k: div.append(length)
        print(div)
        #ADD FUNCTION (check base case!!!!)



    elif (len(previous) > hops and n > hops):
        #eg prev = 13, n = 14 -> start
        #use k = 11 up until 11, then finish delta with remaining k
        #recall
        k = hops
        div = []
        #subdivide the total number into k increments
        while(n-k >= 0):
            div.append(k)
            n -= k
            if n < k:div.append(n)
        print(div)
        for item in div:
            k_hops_2p3, counts = _tri(k_hops_2p3, counts, int(item), k)

    elif (len(previous) > hops and n < hops):
        #eg prev = 13, n = 5 -> start
        #use k = 5 starting at starting city
        k = n
        _, _ = _tri(k_hops_2p3, counts, n, k)

        #options = k_hops_2p3[k-2, int(city_lookup[starting_city]), :]
        # print('third case')
        # print('k is',k)
        # for i in range(n):
        #     #check loop indices
        #     for j in range(k-2,0,-1):
        #         print('Options sum is',options.sum())
        #         if options.sum() != 0:
        #             choice = random.choices(np.arange(k_hops_2p3.shape[1]), weights=options)
        #             print("The next suggested city is", inverse_city_lookup[choice[0]])
        #             k_hops_2p3 = _mask_col(k_hops_2p3, choice[0])
        #             counts = _track_predictions(counts, choice[0])
        #             break
        #         print('looping {} for {} prediction'.format(j, i))
        #         options = k_hops_2p3[j, choice[0], :]
        #     options = k_hops_2p3[k-2, choice[0], :]
    else:
        print("Corner case detected")






def q_2(data, city_count_1, city_totals_1,country_list):

    #previous = input("Please enter a list of previously searched cities, in a list\n")
    #to_search = input("Please enter how many cities you would like to predict ")
    #option = input("Please enter which of the 3 options for question 2")



    city_lookup = dict()
    inverse_city_lookup = dict()
    cities, _ = list(zip(*city_totals_1))
    c = 0
    for item in cities:
        city_lookup[item] = c
        inverse_city_lookup[c] = item
        c += 1
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX']
    previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX',"Stockton CA", 'San Jose CA', 'Seattle WA', 'Portland OR', 'Long Beach CA']
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX',"Stockton CA", 'San Jose CA', 'Seattle WA', 'Portland OR', 'Long Beach CA', 'Los Angeles CA', 'Toronto ON', 'Madison WI']
    to_search = 15
    #V1(city_totals_1, previous, to_search, city_lookup)

    k_hops_2p3 = np.zeros((10, len(cities), len(cities)))
    k_hops_2p2 = np.zeros((len(cities), len(cities)))
    for item in data['cities'].iteritems():
        # row = data.loc[c,'cities'][0].split(', ')
        row = item[1][0].split(', ')
        # print(row)
        # print("row length is ",len(row))
        # print("row number is",c)
        if len(row) == 1:
            pass
        else:
            for i in range(len(row)-1):
                idx = len(row)
                current_city = row[i]
                next_city = row[i+1]
                k_hops_2p2[city_lookup[current_city], city_lookup[next_city]] += 1
                k_hops_2p3[idx - 2 ,city_lookup[current_city] ,city_lookup[next_city]] += 1
    # set rows with all 0's to uniform probability
    k_hops_2p2[np.where(k_hops_2p2.sum(axis=1)==0)] = 1/k_hops_2p2.shape[0]
    k_hops_2p2 = k_hops_2p2/k_hops_2p2.sum(axis=1, keepdims=True)

    k_hops_2p3[np.where(k_hops_2p3.sum(axis=2) == 0)] = 1 / k_hops_2p3.shape[1]
    k_hops_2p3 = k_hops_2p3 / k_hops_2p3.sum(axis=2, keepdims=True)

    #V2(city_totals_1, k_hops_2p2, previous, to_search, city_lookup, inverse_city_lookup)
    V3(city_totals_1, k_hops_2p3, previous, to_search, city_lookup, inverse_city_lookup)





            # doubles
            #         print(len(sub))
                    #k_hops[city_lookup[current_city] ,city_lookup[next_city]] += 1






