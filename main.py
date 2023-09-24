import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

name_to_num = {
    "archer": 0,
    "berserker": 1,
    "cavalry": 2,
    "crossbowman": 3,
    "ensign": 4,
    "footman": 5,
    "knight": 6,
    "lancer": 7,
    "light cavalry": 8,
    "marshall": 9,
    "mercenary": 10,
    "pikeman": 11,
    "royal guard": 12,
    "scout": 13,
    "swordsman": 14,
    "warrior priest": 15
}


def read_data():
    data = pd.read_csv(r"C:\Users\royar\source\repos\WarchestML\data\warchest-game-results.csv")
    col_names = data.columns
    cols_to_remove = [col_name for col_name in col_names if col_name[0] == 'U']
    df = data.drop(columns=cols_to_remove)
    new_columns_data = list()
    columns_names = ['']
    # Loop through each row
    for index, row in df.iterrows():
        # Loop through each column in the row
        vec = [0] * 32
        for column_name, cell_value in row.items():
            col_index = row.index.get_loc(column_name)
            if col_index >= 8:
                continue
            offset = 0 if col_index < 4 else len(name_to_num)
            cell_value = cell_value.lower().strip()
            if cell_value not in name_to_num:
                raise KeyError(f"Why you not in dictionary?! {cell_value}")
            unit_num = name_to_num[cell_value]
            target_col_index = offset + unit_num
            vec[target_col_index] = 1
        new_columns_data.append(vec)
    winners = df['Winner'].apply(lambda x: 0 if x=='S1' else 1)
    new_df = pd.DataFrame(data = new_columns_data, columns=list(range(32)))
    new_df['Winner'] = winners
    return new_df


def cluster_data(df):
    n_clusters = 7
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    return kmeans


if __name__ == '__main__':
    new_df = read_data()
    X = new_df.drop(columns=['Winner'])
    model = cluster_data(X)
    new_df['labels'] = model.labels_
    scores = new_df.groupby('labels')['Winner'].mean()