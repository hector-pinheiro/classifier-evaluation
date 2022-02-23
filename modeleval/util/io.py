import ctypes

import awswrangler
import numpy as np
import pandas as pd


def process_data_types(dataset):
    for col in dataset.columns:
        if dataset[col].dtype in [np.dtype('O').type, np.dtype('S').type]:
            dataset[col] = pd.Series(dataset[col], dtype="category")
        else:
            dataset[col] = dataset[col].astype(ctypes.c_float)


def read_data(data_path):
    df = awswrangler.s3.read_parquet(data_path)
    process_data_types(df)
    return df
