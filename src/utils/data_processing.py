import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocess(path):
    '''
    a function that reorganizes the data and replaces the blank country spaces with 'missing'
    :return: dataframe: processed data
    '''
    data = pd.read_json(path)
    cols = ['userid', 'joining_date', 'country']
    # print(data.iloc[:10,:]['cities'])
    dat = data['user']
    da = list(itertools.chain(*dat))
    df = pd.DataFrame(da)
    o = [df, pd.DataFrame(df[0].tolist()).iloc[:, :3]]
    df2 = pd.concat(o, axis=1).drop(0, axis=1)
    data = data.drop('user', axis=1)
    dff = data.join(df2)
    print("++++++++++++++++++++++++++++++++")
    dff['country'] = dff['country'].replace('', 'missing')
    dff = dff.sort_values(by='country')
    # print(dff.head())
    return dff


def create_cityinfo(data):
    '''
    a function that amalgamates all the cities visited per country
    :return: dict: count of cities visited, dict: cities visited per country
    '''
    city_count = dict()
    for it in data['country'].unique():
        city_count[it] = {}
    city_totals = dict()
    print(city_count)
    c = 0
    for item in data['cities'].iteritems():
        for i in range(len(item[1][0].split(', '))):
            if item[1][0].split(', ')[i] in city_count[data.loc[c, 'country']]:
                city_count[data.loc[c, 'country']][item[1][0].split(', ')[i]] += 1
            else:
                city_count[data.loc[c, 'country']][item[1][0].split(', ')[i]] = 1

            if item[1][0].split(', ')[i] in city_totals.keys():
                city_totals[item[1][0].split(', ')[i]] += 1
            else:
                city_totals[item[1][0].split(', ')[i]] = 1
        c += 1

    city_keys = set(city_totals.keys())
    city_count_arr = []
    # add additional 0's for all the entries that aren't in country j that are in the total
    country_list = []
    for item in city_count:
        country_list.append(item)
        country_keys = set(city_count[item].keys())
        diff = city_keys - country_keys
        for city in diff:
            city_count[item][city] = 0
        # city_count_arr.append(np.asarray([value for (key, value) in sorted(city_count[item].items())]))
    # city_count_arr = np.asarray(city_count_arr).squeeze()

    city_totals_l = sorted(list(zip(city_totals.keys(), city_totals.values())), key=lambda x: x[1], reverse=True)
    city_count_l = dict()
    for item in city_count:
        city_count_l[item] = sorted(list(zip(city_count[item].values(), city_count[item].keys())), key=lambda x: x[1])
    return city_count_l, city_totals_l, city_count, city_totals, country_list
    #return city_totals, city_count
