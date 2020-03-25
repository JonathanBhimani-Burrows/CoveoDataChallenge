import pandas as pd
import itertools
import matplotlib.pyplot as plt

def read_prev(path):
    '''
    a function that reads the previous cities from a text file
    :param path: path to find txt file
    :return: the list of cities
    '''
    f = open(path, "r")
    line = f.readline()
    if len(line) == 0:
        l = []
    else:
        l = line.split(",")
    return l


def preprocess(path):
    '''
    a function that reorganizes the data and replaces the blank country spaces with 'missing'
    :return: dataframe: processed data
    '''
    data = pd.read_json(path)
    dat = data['user']
    da = list(itertools.chain(*dat))
    df = pd.DataFrame(da)
    df1 = [df, pd.DataFrame(df[0].tolist()).iloc[:, :3]]
    df2 = pd.concat(df1, axis=1).drop(0, axis=1)
    data = data.drop('user', axis=1)
    dff = data.join(df2)
    dff['country'] = dff['country'].replace('', 'missing')
    dff = dff.sort_values(by='country')
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
    c = 0
    #creates city counts (total aggregate and also per country)
    for item in data['cities'].iteritems():
        row = item[1][0].split(', ')
        for i in range(len(row)):
            if row[i] in city_count[data.loc[c, 'country']]:
                city_count[data.loc[c, 'country']][row[i]] += 1
            else:
                city_count[data.loc[c, 'country']][row[i]] = 1

            if row[i] in city_totals.keys():
                city_totals[row[i]] += 1
            else:
                city_totals[row[i]] = 1
        c += 1

    city_keys = set(city_totals.keys())
    # add additional 0's for all the entries that aren't in country j that are in the total
    country_list = []
    for item in city_count:
        country_list.append(item)
        country_keys = set(city_count[item].keys())
        diff = city_keys - country_keys
        for city in diff:
            city_count[item][city] = 0
    city_totals_l = sorted(list(zip(city_totals.keys(), city_totals.values())), key=lambda x: x[1], reverse=True)
    city_count_l = dict()
    for item in city_count:
        city_count_l[item] = sorted(list(zip(city_count[item].values(), city_count[item].keys())), key=lambda x: x[1])
    return city_count_l, city_totals_l, city_count, city_totals, country_list



def plot_features(features):
    '''
    a function used to plot the cities
    :param features: features to plot
    :return: 0
    '''
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    for i in range(len(features)):
        ax.scatter(features[i,0], features[i,1], color='b')
        ax.set_title("PCA Components")
    #plt.legend(country_list)
    plt.show()