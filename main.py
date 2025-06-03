import Interpolation
import RandomForest
import DataCleaning
import pandas as pd
import time

def main():

    data = DataCleaning.DataCleaning("fire20062024.xlsx") #, "fire19962005.csv", "fire19831995.csv"  

    #Interpolates missing data with Linear and Logistic regression
    dataInterpolated = data.dfInterpolated
    
    #Imputes missing data with mean and mode feature values  
    # dataImputed = data.dfImputed

    #Drops NA values from data
    dfDropNull = data.dfDropNull

    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0])
    RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat=True, testData = dfDropNull)
    


if __name__ == "__main__":
    main()
