import argparse
import pandas as pd
import itertools
from utils.data_processing import *
from questions.question1 import *
from questions.question2 import *
from questions.question3 import *
from questions.question4 import *





def main():
    #read argparse arguments
    data_path = args.data_path
    question = int(args.question)
    previous = args.path_to_previous
    to_search = int(args.num_predictions)
    country = args.country
    previous = read_prev(previous)
    #data preprocesing
    data = preprocess(data_path)
    #generate different city information
    city_count_l, city_totals_l, city_count, city_totals, country_list = create_cityinfo(data)
    if question == 1:
        q_1(city_count_l, city_totals_l,country_list)
    else:

        if question == 2:
            _,_,_ = q_2(data, city_count_l, city_totals_l,country_list, previous, to_search)
        elif question == 3:
            _ = q_3(data, city_count_l, city_totals_l,country_list, previous, to_search, country)
        elif question == 4:
            predictions_V1, predictions_V2, predictions_V3  = q_2(data, city_count_l, city_totals_l,
                                                                                country_list,previous, to_search)
            predictions_Q3 = q_3(data, city_count_l, city_totals_l, country_list, previous, to_search, country)


            q_4(data, city_count_l, city_totals_l,country_list, predictions_V1, predictions_V2, predictions_V3,
                        predictions_Q3)
        else:
            print('Not Implemented')


if __name__ == '__main__':
    print("Sample input: python main.py $data_path$ 1")
    print("Sample input: python main.py $data_path$ 2 --path_to_previous $path_to_previous$ --num_predictions 3")
    print("Sample input: python main.py $data_path$ 3 --path_to_previous $path_to_previous$ --num_predictions 3 --country UK " )
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("question")
    parser.add_argument("--path_to_previous", help="Path to sequence of previous countries. Required for questions 2-4")
    parser.add_argument("--num_predictions", help="How many predictions to make. Required for questions 2-4", type=int)
    parser.add_argument("--country", help="Country of user. Required for questions 3 and 4", type=str)
    args = parser.parse_args()
    main()