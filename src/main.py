import argparse
import pandas as pd
import itertools
from utils.data_processing import read_prev, preprocess, create_cityinfo, plot_features
from questions.question1 import q_1
from questions.question2 import q_2
from questions.question3 import q_3
from questions.question4 import q_4


def main():
    '''
    main function for the data challenge
    :return: 0
    '''
    #read argparse arguments
    data_path = args.data_path
    question = int(args.question)
    #data preprocesing
    data = preprocess(data_path)
    #generate different city information
    city_count_l, city_totals_l, city_count, city_totals, country_list = create_cityinfo(data)
    if question == 1:
        q_1(city_count_l,country_list)
    else:
        #process additional arguments
        previous = args.path_to_previous
        to_search = int(args.num_predictions)
        country = args.country
        previous = read_prev(previous)
        # run question 2
        if question == 2:
            _,_,_,_,_ = q_2(data, city_totals_l, previous, to_search)
        #run question 3
        elif question == 3:
            _ = q_3(data, city_count_l, city_totals_l,country_list, previous, to_search, country)
        #run question 4
        elif question == 4:
            predictions_V1, predictions_V2, predictions_V3, city_lookup, inverse_city_lookup  = q_2(data, city_totals_l,
                                                                                                previous, to_search)
            predictions_V4 = q_3(data, city_count_l, city_totals_l, country_list, previous, to_search, country)
            preds = (predictions_V1, predictions_V2,predictions_V3, predictions_V4)
            q_4(data, city_totals_l, preds, previous, to_search,city_lookup, inverse_city_lookup)
        else:
            print('Not Implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("question")
    parser.add_argument("--path_to_previous", help="Path to sequence of previous countries. Required for questions 2-4")
    parser.add_argument("--num_predictions", help="How many predictions to make. Required for questions 2-4", type=int)
    parser.add_argument("--country", help="Country of user. Required for questions 3 and 4", type=str)
    args = parser.parse_args()
    main()