from zlib import crc32

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DEFAULT_SCORES_FILE_PATH = r'..\..\Output\scores.txt'


class Data:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_data(self):
        return pd.read_csv(self.csv_path)

    def check_crc_test_data(self, identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2 ** 3

    def split_train_test_by_id(self, data, test_ratio, id_column):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.check_crc_test_data(id_, test_ratio))
        return data.loc[~in_test_set], data.loc[in_test_set]

    def replacing_NA(self, df, column_name, custom_value=0):
        if custom_value != 0:
            df[column_name].fillna(custom_value, inplace=True)
        else:
            # check median value from dataframe
            median = df[column_name].median()
            df[column_name].fillna(median, inplace=True)

    def display_scores(self, scores, custom_start_print="Wyniki:"):
        """funkcja zwraca wyniki poprzez sprawdzian krzyżowy dla algorytmu
        scores to obiekt scores utworzony przez klasę cross_val_scores"""
        print(custom_start_print, scores)
        print("Średnia:", scores.mean())
        print("Odchylenie standardowe:", scores.std())

    def saving_scores_to_file(self, scores, file_path=DEFAULT_SCORES_FILE_PATH,
                              custom_start_print="Wyniki dla modelu\n"):
        with open(file_path, "a") as file:
            file.write(custom_start_print)
            # saving tab to file
            for i in scores:
                file.write(f"{i}\n")


if __name__ == '__main__':
    data = Data("../../Datasets/gpa_study_hours.csv")
    fetch_data = data.load_data()
    # for all columns in dataframe fillna with median value

    print(fetch_data.head())

    # fetch_data["gpa"].hist(bins=50,figsize=[20,15])
    fetch_data["study_hours"].hist(bins=50, figsize=[20, 15])
    plt.show()
    # train_set, test_set = data.split_train_test_by_id(fetch_data,0.2,"Id")
    # train_set, test_set = train_test_split(fetch_data, test_size=0.2, random_state=42)
    # # get histograms for test data
    # train_set["discount"].hist()
    # test_set["discount"].hist()
    # plt.show()
