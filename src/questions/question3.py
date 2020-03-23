import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.data_processing import *
import random
from questions.question2 import V2


def q_3(data, city_count_l, city_totals_l,country_list, previous, to_search, country):
    '''
    a function that runs a slighly modified version of question 2
    :param data: raw data
    :param city_count_l: list count of cities per country
    :param city_totals_l: total counts of cities
    :param country_list: list of countries
    :return: 0
    '''
    print("++++++++++++++Question 3+++++++++++++++++++")
    predictions = []
    city_lookup = dict()
    inverse_city_lookup = dict()
    cities, _ = list(zip(*city_totals_l))
    print("city totals in 3 is", len(city_totals_l))
    c = 0
    for item in cities:
        city_lookup[item] = c
        inverse_city_lookup[c] = item
        c += 1
    c = 0
    country_lookup = dict()
    for item in country_list:
        country_lookup[item] = c
        c += 1

    k_hops_per_country = create_hops(data, cities, city_lookup, countries=country_lookup)
    #previous = []
    #previous = ['New York NY', 'Montreal QC', 'Minneapolis MN', 'Austin TX',"Stockton CA", 'San Jose CA', 'Seattle WA', 'Portland OR', 'Long Beach CA', 'Los Angeles CA', 'Toronto ON', 'Madison WI']
    #to_search = 10
    #country = 'UK'
    country_idx = country_lookup[country]
    country_hop = k_hops_per_country[country_idx, :,:]
    #reusing the same code as question 2.2
    predictions = V2(city_totals_l, country_hop, previous, to_search, city_lookup, inverse_city_lookup)
    return predictions