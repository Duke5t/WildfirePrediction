import Interpolation
import RandomForest
import XGBoost
import DataCleaning
import pandas as pd
import EDA
import LightGBM

#Notes:
#IF we use test data in a model, we must use either interpolated or imputed data since those dataFrames exclude test data

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
    testData = data.dfTestData

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
    # EDA.EDA(data.df)
    #END EDA





    #RandomForest
    # RFModel_IntFireSpread = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    # RFModel_ImpFireSpread = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)

    # RFModel_IntSizeClass = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    # RFModel_ImpSizeClass = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    

    # XGBoost - uses data with null values
    # XGBModel_FireSpread = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    # XGBModel_SizeClass = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)

    # XGBModel_SizeClass = XGBoost.XGBoost(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)


    #LightGBM
    LGBM_FireSpread = LightGBM.LightGBM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    LGBM_SizeClass = LightGBM.LightGBM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)






if __name__ == "__main__":
    main()
