import DataCleaning
import pandas as pd
import time

def main():

    data = DataCleaning.DataCleaning("fire20062024.xlsx") 

    print(data.df.isnull().sum())

if __name__ == "__main__":
    main()
