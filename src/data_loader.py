import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
path='data/sp500.csv'
def load_and_preprocess_data(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = df.columns.str.strip()

    print("Columns after stripping whitespace:")
    print(df.columns.tolist())

    df.rename(columns={
       'Price': 'Date',
       'Unnamed: 1': 'Close',
       'Unnamed: 2': 'High',
       'Unnamed: 3': 'Low',
       'Unnamed: 4': 'Open',
       'Unnamed: 5': 'Volume'
        }, inplace=True)
    print("Columns after stripping whitespace:")
    print(df.columns.tolist())
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')
      
    df['Date']=pd.to_datetime(df['Date'], errors='coerce')

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df




