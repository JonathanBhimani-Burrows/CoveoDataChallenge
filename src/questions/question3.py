from utils.matrix_processing import create_hops
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
    city_lookup = dict()
    inverse_city_lookup = dict()
    cities, _ = list(zip(*city_totals_l))
    c = 0
    #create lookups
    for item in cities:
        city_lookup[item] = c
        inverse_city_lookup[c] = item
        c += 1
    c = 0
    country_lookup = dict()
    for item in country_list:
        country_lookup[item] = c
        c += 1
    # create country based hop table
    k_hops_per_country = create_hops(data, cities, city_lookup, countries=country_lookup)
    country_idx = country_lookup[country]
    #slice this hop table and use with code in question 2.2
    country_hop = k_hops_per_country[country_idx, :,:]
    predictions = V2(city_totals_l, country_hop, previous, to_search, city_lookup, inverse_city_lookup)
    return predictions