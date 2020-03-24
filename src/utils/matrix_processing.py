import numpy as np
from utils.data_processing import *


def track_predictions(counts, idx):
    '''
    a function that tracks the cities visited if the probabilities of a row go to 0, so an independent
    high probabily option can be chosen
    :param counts: counts of each city
    :param idx: chosen city index
    :return: counts with city index set to 0
    '''
    counts[idx] = 0
    return counts

def mask_col(mat, col,val):
    '''
    a function that masks the columns that have already been visited
    :param col: column to be masked
    :return: hop matrix with column masked
    '''
    if mat.ndim == 2:
        mat[:, col] = val
    elif mat.ndim == 3:
        mat[:, :, col] = val
    else:
        raise ValueError('Wrong matrix dimension')
    return mat


def mask_previous(hops, previous, counts, cities, city_lookup, val):
    '''
    a function that masks previous visited cities so they aren't visited again
    :param hops: hop matrix to mask
    :param previous: list of previously visited cities
    :param counts: list of number of times each city was visited
    :param cities: list of cities
    :param city_lookup: a lookup table for the cities
    :param val: the value to mask with
    :return: masked hop matrix, updated counts and the starting (most recently visited) city
    '''
    if len(previous) != 0:
        for item in previous:
            hops = mask_col(hops, int(city_lookup[item]),val)
            counts = track_predictions(counts, int(city_lookup[item]))
        starting_city = previous[-1]
    else:
        starting_city = cities[np.argmax(counts)]
        counts = track_predictions(counts, np.argmax(counts))
    return hops, counts, starting_city

def create_hops(data, cities, city_lookup, **kwargs):
    '''
    a function that creates the hop matrix from the raw data
    :param data: raw data
    :param cities: list of cities
    :param city_lookup: lookup table for cities
    :param kwargs: hop depth - number of cities in the searches, countries - number of countries
    :return: creates the hop matrix
    '''
    if 'hop_depth' in kwargs:
        hops = np.zeros((kwargs['hop_depth'], len(cities), len(cities)))
    elif 'countries' in kwargs:
        hops = np.zeros((len(kwargs['countries']), len(cities), len(cities)))
        country_lookup = kwargs['countries']
    else:
        hops = np.zeros((len(cities), len(cities)))

    for item in data['cities'].iteritems():
        row = item[1][0].split(', ')
        if len(row) == 1:
            pass
        else:
            for i in range(len(row)-1):
                idx = len(row)
                current_city = row[i]
                next_city = row[i+1]
                if 'hop_depth' in kwargs:
                    hops[idx - 2 ,city_lookup[current_city] ,city_lookup[next_city]] += 1
                elif 'countries' in kwargs:
                    hops[country_lookup[data.loc[item[0],'country']], city_lookup[current_city],
                         city_lookup[next_city]] += 1
                else:
                    hops[city_lookup[current_city], city_lookup[next_city]] += 1
    # set rows with all 0's to uniform probability
    if kwargs:
        hops[np.where(hops.sum(axis=2) == 0)] = 1 / hops.shape[1]
        hops = hops / hops.sum(axis=2, keepdims=True)
    else:
        hops[np.where(hops.sum(axis=1) == 0)] = 1/hops.shape[1]
        hops = hops/hops.sum(axis=1, keepdims=True)
    return hops