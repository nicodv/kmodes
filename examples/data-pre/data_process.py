# -*- coding: UTF-8 -*-
"""
@Project ：kmodes
@File    ：Large_soybean.py
@Author  ：Fangqi Nie
@Date    ：2022/10/30 15:39
@Contact :   kristinaNFQ@163.com
"""

import pandas as pd
import numpy as np
# 将文字编码
from sklearn.preprocessing import LabelEncoder


def large_soybean(data):
    df_notnull = data.dropna()
    df = df_notnull
    df.Column1 = LabelEncoder().fit_transform(df.Column1)
    data = np.array(df)
    data_final = data[~np.any(data == '?', axis=1)]
    data_final = pd.DataFrame(data_final)
    data_final.to_csv('Encode_soybean_large.csv', index=False, header=0)


def Mushroom(data):
    df_notnull = data.dropna()
    df = df_notnull
    for i in range(1, 24):
        df[f"Column{i}"] = LabelEncoder().fit_transform(df[f"Column{i}"])
    df.to_csv('Encode_mushroom.csv', index=False, header=0)


def Voting(data):
    data = data.dropna()
    data_final = data[~np.any(data == '?', axis=1)]
    df = data_final
    for i in range(1, 18):
        df[f"Column{i}"] = LabelEncoder().fit_transform(df[f"Column{i}"])
    df.to_csv('Encode_Voting.csv', index=False, header=0)