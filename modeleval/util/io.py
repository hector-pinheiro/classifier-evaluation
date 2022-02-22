import ctypes

import awswrangler
import numpy as np
import pandas as pd


def process_data_types(dataset, columns):
    for col in columns:
        if dataset[col].dtype in [np.dtype('O').type, np.dtype('S').type]:
            dataset[col] = pd.Series(dataset[col], dtype="category")
        else:
            dataset[col] = dataset[col].astype(ctypes.c_float)


def read_df(df_path, features, labels):
    df = awswrangler.s3.read_parquet(df_path)
    process_data_types(df, features + labels)
    return df
