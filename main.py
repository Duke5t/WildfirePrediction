import Interpolation
import RandomForest
import XGBoost
import DataCleaning
import pandas as pd
import time

def main():
    data = DataCleaning.DataCleaning("fire20062024.xlsx") #, "fire19962005.csv", "fire19831995.csv"  

    ##TEST Boolean used to expedite debuging of models. Triggers all other class test booleans 
    TEST = False #TOGGLE ME 
    if TEST:
        data.TEST = True


    
    data.setQuantitativeFeatures(["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED"]) #, "DISTANCE_FROM_WATER_SOURCE"
    data.setCategoricalFeatures(["FUEL_TYPE", "TRUE_CAUSE", "DETECTION_AGENT", "DETECTION_AGENT_TYPE", "FIRE_POSITION_ON_SLOPE", "WEATHER_CONDITIONS_OVER_FIRE", "GENERAL_CAUSE"])
    data.setPredictionFeatures(["FIRE_SPREAD_RATE", "SIZE_CLASS"])

    data.createDataFrames()
    dataRaw = data.df

    #Interpolates missing data with Linear and Logistic regression
    if not TEST:
        dataInterpolated = data.dfInterpolated
        #Imputes missing data with mean and mode feature values  
        dataImputed = data.dfImputed
        # Drops NA values from data
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
