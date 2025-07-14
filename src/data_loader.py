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
    
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    df['Volume_change']=df['Volume'].pct_change()
    delta = df['Close'].diff()
    gain=delta.clip(lower=0)
    loss= -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs =avg_gain/(avg_loss+1e-10)
    df['RSI_14']=100-(100/(1+rs))
    print(df.columns.tolist())
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df




