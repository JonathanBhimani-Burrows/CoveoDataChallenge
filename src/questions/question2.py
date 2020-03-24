import numpy as np
from utils.matrix_processing import mask_col, mask_previous, track_predictions, create_hops
import random


def _choose_prediction(hops, options, counts, inverse_city_lookup,predictions):
    '''
    a helper function that chooses the next predicted city and runs the mask col helper function
    :param hops: hop matrix
    :param options: the list of possible probabilties for the algorithm to choose
    :param counts: counts of each city
    :param inverse_city_lookup: inverse lookup table
    :param predictions: saved list of predictions
    :return: updated hop matrix, updated options, updated counts, choice of next ciyy and updated list of predictions
    '''
    choice = random.choices(np.arange(hops.shape[1]), weights=options)
    print("The next suggested city is", inverse_city_lookup[choice[0]])
    predictions.append(inverse_city_lookup[choice[0]])
    hops = mask_col(hops, choice[0], 0)
    counts = track_predictions(counts, choice[0])
    return hops, options, counts, choice, predictions

def V1(city_totals_l, previous, n, city_lookup):
    '''
    a function that prints suggested city (or cities) based on an independence assumption
    :param city_totals_l: total times a city has been visited
    :param previous: list of previously visited cities
    :param n: number of cities to suggest
    :param city_lookup: lookup table for city indices
    :return: list of predictions
    '''
    predictions = []
    if len(previous) > 0:
        cities, counts = list(zip(*city_totals_l))
        counts = list(counts)
        for item in previous:
            counts[int(city_lookup[item])] = 0
    counts = np.array(counts)
    for i in range(n):
        print("The next suggested city is",cities[np.argmax(counts)])
        counts[np.argmax(counts)] = 0
        predictions.append(cities[np.argmax(counts)])
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
    :return: list of predictions
    '''
    predictions = []
    cities, counts = list(zip(*city_totals_l))
    counts = np.asarray(counts)
    #mask cities already searched
    k_hops_2p2, counts, starting_city = mask_previous(k_hops_2p2, previous, counts, cities, city_lookup, 0)

    #make suggestions
    options = k_hops_2p2[int(city_lookup[starting_city]), :]
    c = 0
    #loop over all cities that we need to predict
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
                counts = track_predictions(counts, choice)
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
    :return: list of predictions
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
            print("k is",k)
            for j in range(k - 2, -1, -1):
                if flag:
                    options = k_hops_2p3[j, choice[0], :]
                flag = True
                print('options sum is', options.sum())
                if options.sum() != 0:
                    k_hops_2p3, options, counts, choice, predictions = _choose_prediction(k_hops_2p3, options, counts,
                                                                                    inverse_city_lookup,predictions)
                    break
                elif options.sum() == 0:
                    choice = [np.argmax(counts)]
                if j == 0:
                    choice = [0]
                print("The next suggested city is", inverse_city_lookup[choice[0]])
                predictions.append(inverse_city_lookup[choice[0]])
                counts = track_predictions(counts, choice[0])
                k_hops_2p3 = mask_col(k_hops_2p3, choice[0], 0)
            flag = False
            options = k_hops_2p3[k - 2, choice[0], :]
        return k_hops_2p3, counts, predictions

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
    ############ Main Function Starts Here ##########
    predictions = []
    cities, counts = list(zip(*city_totals_l))
    counts = np.asarray(counts)

    #mask cities already searched
    k_hops_2p3, counts, starting_city = mask_previous(k_hops_2p3, previous, counts, cities, city_lookup, 0)
    length = len(previous) + n
    hops = k_hops_2p3.shape[0] + 1
    if length <= hops:
        print("condition 1")
        #eg prev = 4, n = 3 -> length = 7
        # use k = 7 starting at starting city
        k = length
        _, _, predictions = _run_down_hops(k_hops_2p3, counts,cities, n, k,inverse_city_lookup, predictions)
    elif (len(previous) < hops and length >= hops ):
        print("condition 2")
        # eg prev = 9, n = 5 -> length = 14 or prev = 9, n = 15 -> length = 24
        # use k = 11 up until 11, then finish delta with remaining k
        k = hops

        # subdivide the total number into k length increments
        div = _generate_subdivisions(length, k)
        print("Div is", div)
        div[0] -= len(previous)
        for item in div:
            k_hops_2p3, counts, predictions = _run_down_hops(k_hops_2p3, counts,cities, int(item), k,
                                                             inverse_city_lookup, predictions)
    elif (len(previous) > hops and n > hops):
        print("condition 3")
        #eg prev = 13, n = 14 -> start
        #use k = 11 up until 11, then finish delta with remaining k
        k = hops
        #subdivide the total number into k increments
        div = _generate_subdivisions(n, k)
        for item in div:
            k_hops_2p3, counts, predictions = _run_down_hops(k_hops_2p3, counts,cities, int(item), k,
                                                             inverse_city_lookup, predictions)
    elif (len(previous) > hops and n < hops):
        print("condition 4")
        #eg prev = 13, n = 5 -> start
        #use k = 5 starting at starting city
        if n < 3:
            k = 2
        else:
            k = n
        _, _, predictions = _run_down_hops(k_hops_2p3, counts,cities, n, k, inverse_city_lookup, predictions)
    else:
        print("Corner case detected")
    return predictions


def q_2(data, city_totals_l, previous, to_search):
    '''
    a function that runs question 2
    :param data: raw data
    :param city_totals_l: total aggregate of cities visited per country
    :param previous: previous cities visited (by user)
    :param to_search: number of cities to search
    :return: predictions
    '''
    print("++++++++++++++Question 2+++++++++++++++++++")
    #create lookup and inverse city lookups
    city_lookup = dict()
    inverse_city_lookup = dict()
    cities, counts = list(zip(*city_totals_l))
    c = 0
    for item in cities:
        city_lookup[item] = c
        inverse_city_lookup[c] = item
        c += 1
    hop_depth = 10
    print("++++++++++++++Question 2.1+++++++++++++++++++")
    predictions_V1 = V1(city_totals_l, previous, to_search, city_lookup)

    print("++++++++++++++Question 2.2+++++++++++++++++++")
    k_hops_2p2 = create_hops(data, cities, city_lookup)
    predictions_V2 = V2(city_totals_l, k_hops_2p2, previous, to_search, city_lookup, inverse_city_lookup)

    print("++++++++++++++Question 2.3+++++++++++++++++++")
    k_hops_2p3 = create_hops(data, cities, city_lookup, hop_depth=hop_depth)
    predictions_V3 = V3(city_totals_l, k_hops_2p3, previous, to_search, city_lookup, inverse_city_lookup)
    return predictions_V1, predictions_V2, predictions_V3, city_lookup, inverse_city_lookup








