import os

import pandas
import pandas as pd
import pandas_ta as ta
import glob
from typing import Union, List


def concatenate_CSVs():
    """
    Function that was used to concatenate all the downloaded csvs into one giant file
    :return:
    """
    csvs_path = "BTC Minute CSVs"
    all_csvs = glob.glob(csvs_path + "/*.csv")
    csvs = []
    for csv in all_csvs:
        df = pd.read_csv(csv, index_col=None, header=0)
        csvs.append(df)

    csvs.reverse()
    frame = pd.concat(csvs, axis=0, ignore_index=True)
    save_path = "BTC Minute CSVs/gemini_BTCUSD_all_1min.csv"
    frame.to_csv(save_path, index=False)


def reverse_rows():
    """
    Function that was used to reverse the rows of the giant csv (it was from 2021 to 2015)
    :return:
    """
    path_to_csv = "BTC Minute CSVs/gemini_BTCUSD_all_1min.csv"
    df = pd.read_csv(path_to_csv, index_col=None, header=0)
    df = df.reindex(index=df.index[::-1])
    save_path = "BTC Minute CSVs/gemini_BTCUSD_all_1min_reversed.csv"
    df.to_csv(save_path, index=False)


def calculate_rsi(df: pd.DataFrame, length: int = 14):
    """
    Wrapper function for RSI
    :param df:
    :param length:
    :return:
    """
    if not isinstance(length, int):
        raise Exception("length must be an integer!")

    df[f'RSI_{length}'] = ta.rsi(df['Close'], length=length)


def calculate_sma(df: pd.DataFrame, length: int = 50):
    """
    Wrapper function for sma
    :param df:
    :param length:
    :return:
    """
    if not isinstance(length, int):
        raise Exception("length must be an integer!")
    df[f'SMA_{length}'] = ta.sma(df['Close'], length=length)


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Wrapper function for MACD
    :param df:
    :param fast:
    :param slow:
    :param signal:
    :return:
    """
    if not isinstance(fast, int):
        raise Exception("fast must be an integer!")
    if not isinstance(slow, int):
        raise Exception("slow must be an integer!")
    if not isinstance(signal, int):
        raise Exception("signal must be an integer!")
    df[f'MACD_{fast}_{slow}_{signal}'] = ta.macd(close=df['Close'], fast=fast, slow=slow, signal=signal)[
        f'MACD_{fast}_{slow}_{signal}']


def main():
    # load the data
    # BTCUSD_all_simple.parquet contains all CSV files merged into one, without any processing done on them
    load_path = os.path.join('BTC Minute CSVs', 'BTCUSD_all_simple.parquet')
    df = pd.read_parquet(load_path, engine='fastparquet')

    # do what you wanna do
    calculate_rsi(df, 14)
    calculate_sma(df)
    calculate_macd(df)

    # saving the new data
    save_path = os.path.join('BTC Minute CSVs', 'BTCUSER_all_processed.parquet')
    df.to_parquet(save_path, engine='fastparquet')

    # if you want to check the values, uncomment the 2 lines bellow to save them in a csv file
    # csv_path = os.path.join('BTC Minute CSVs', 'BTCUSER_all_processed.csv')
    # df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
