import Interpolation
import RandomForest1
import RandomForest
import XGBoost
import DataCleaning
import pandas as pd
import time

def main():

    TEST = False
    
    data = DataCleaning.DataCleaning("fire20062024.xlsx") #, "fire19962005.csv", "fire19831995.csv"  
    dataRaw = data.df

    #Interpolates missing data with Linear and Logistic regression
    if not TEST:
        dataInterpolated = data.dfInterpolated

    #Imputes missing data with mean and mode feature values  
    if not TEST:
        dataImputed = data.dfImputed

    # Drops NA values from data
    if not TEST:
        dfDropNull = data.dfDropNull

    if TEST:
        dataInterpolated = data.dfDropNull
        dataImputed = data.dfDropNull
        dfDropNull = data.dfDropNull


        
    #EDA

    #END EDA

    #RandomForest
    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = dfDropNull)
    RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = dfDropNull)
    RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    

    # XGBoost - uses data with null values
    XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)



if __name__ == "__main__":
    main()
