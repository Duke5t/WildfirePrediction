import Interpolation
import RandomForest
import XGBoost
import DataCleaning
import pandas as pd
import EDA
import LightGBM
import SVM


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
        dataDropNull = data.dfDropNull

    if TEST:
        dataInterpolated = data.dfDropNull
        dataImputed = data.dfDropNull
        dataDropNull = data.dfDropNull
        
    #EDA
    # EDA.EDA(d/ata.df)
    #END EDA

    #RandomForest
    # print("\n\nRF Interpolated")
    # RF_IntFireSpread = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    print("\n\nRF Imputed")
    RF_ImpFireSpread = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    # print("\n\nRF DropNull")
    # RF_NoNullFireSpread = RandomForest.RandomForest(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    
    # # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)

    # print("\n\nRF Interpolated")
    # RF_IntSizeClass = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    print("\n\nRF Imputed")
    RF_ImpSizeClass = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    # print("\n\nRF NoNull")
    # RF_NoNullSizeClass = RandomForest.RandomForest(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    # RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    

    # XGBoost - uses data with null values (dont use special test data here)
    XGB_FireSpread = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    XGB_SizeClass = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)


    #LightGBM
    LGBM_FireSpread = LightGBM.LightGBM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    LGBM_SizeClass = LightGBM.LightGBM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)

    #SVM
    SVM_FireSpread = SVM.SVM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    SVM_SizeClass = SVM.SVM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)


    #CREATES CSV OF ALL MODEL RESULTS
    result = pd.concat(
        [RF_ImpFireSpread.results, RF_ImpSizeClass.results,
         LGBM_FireSpread.results, LGBM_SizeClass.results, 
         XGB_FireSpread.results, XGB_SizeClass.results,
         SVM_FireSpread.results, SVM_SizeClass.results
        ], axis=0)
    result.to_csv("ModelResults.csv", float_format='%.4f')


if __name__ == "__main__":
    main()
