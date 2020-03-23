import argparse
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.data_processing import *


def plot_countries(features,country_list):
    '''
    a function that plots the pca components of the vectors
    :param features: pca features for each country
    :param country_list: list of countries
    :return: 0
    '''
    colours = ['r','b','y','g','c','m','k']
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    for i in range(len(features)):
        ax.scatter(features[i,0], features[i,1], color=colours[i])
        ax.set_title("PCA Components")
    plt.legend(country_list)
    #plt.show()


def cosine_sim(city_count, country_list):
    '''
    a function that calculates the cosine similarties between vectors
    :param city_count: counts of cities per country
    :param country_list: list of countries
    :return: 0
    '''
    num_cities = city_count.shape[0]
    similarities = np.zeros((num_cities, num_cities))
    print(similarities.shape)
    max = 0
    min = 10
    for i in range(num_cities):
        for j in range(i, num_cities):
            similarities[i][j] = cosine_similarity(city_count[i].reshape(1,-1), city_count[j].reshape(1,-1))
            if similarities[i][j] > max:
                max = similarities[i][j]
            if similarities[i][j] < min:
                min = similarities[i][j]
    #normalize the values
    similarities = (similarities - min)/ (max - min)
    similarities[similarities < 0] = 0
    print(country_list)
    print(similarities)
    print("Max similarity for the missing country is",np.sort(similarities[:,6])[len(similarities[:,6])-2])




def q_1(city_count_l, city_totals_l, country_list):
    '''
    a function that uses city information to generate cosine similarties/pca components
    :param city_count_l: counts of cities per country
    :param city_totals_l: total number of cities
    :param country_list: list of countries
    :return: 0
    '''
    print("++++++++++++++Question 1+++++++++++++++++++")
    features = []
    for item in city_count_l:
        features.append(list(zip(*city_count_l[item]))[0])
    #runs cosine similarities
    features = np.asarray(features)
    cosine_sim(features, country_list)
    # run PCA and plot countries
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    plot_countries(pca_result,country_list)

