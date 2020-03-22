import argparse
import pandas as pd
import itertools
from utils.data_processing import *
from questions.question1 import *
from questions.question2 import *





def main():
    data_path = args.data_path
    question = int(args.question)
    data = preprocess(data_path)
    city_count_l, city_totals_l, city_count, city_totals, country_list = create_cityinfo(data)
    if question == 1:
        q_1(city_count_l, city_totals_l,country_list)
    elif question == 2:
        q_2(data, city_count_l, city_totals_l,country_list)
    else:
        print('Not Implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("question")
    args = parser.parse_args()
    main()