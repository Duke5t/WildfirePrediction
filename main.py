import Imputation
import DataCleaning
import pandas as pd
import time

def main():

    data = DataCleaning.DataCleaning("fire20062024.xlsx") #, "fire19962005.csv", "fire19831995.csv"  

    dataWImp = Imputation.Imputation(data, data.quantFeat, data.catFeat)

if __name__ == "__main__":
    main()
