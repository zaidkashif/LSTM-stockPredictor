import yfinance as yf
import pandas as pd
# df = yf.download('^GSPC', start = '2010-01-01', end='2025-07-10')
# df.to_csv('data/sp500.csv')

df = pd.read_csv('data/sp500.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())