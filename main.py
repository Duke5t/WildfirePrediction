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
    # data = DataCleaning.DataCleaning("AlbertaWildfireData.xlsx") #, "fire19962005.csv", "fire19831995.csv"  

    ##TEST Boolean used to expedite debuging of models. Triggers all other class test booleans 
    TEST = False #TOGGLE ME 
    if TEST:
        data.TEST = True


    
    data.setQuantitativeFeatures(["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED", "LATITUDE", "LONGITUDE"]) 
    data.setCategoricalFeatures(["FUEL_TYPE", "GENERAL_CAUSE", "FIRE_POSITION_ON_SLOPE", 
                                 "WEATHER_CONDITIONS_OVER_FIRE", "DETECTION_AGENT",
                                  "FIRE_START_DATE", "WIND_DIRECTION"]) #"TRUE_CAUSE", "DETECTION_AGENT_TYPE", 
    data.setPredictionFeatures(["FIRE_SPREAD_RATE", "SIZE_CLASS"])

    #EDA
    EDA.EDA(data.df)
    #END EDA


    data.createDataFrames()
    # data.featureEngineering()
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
        

    #RandomForest
    RF_IntFireSpread = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    # RF_ImpFireSpread = RandomForest.RandomForest(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    RF_ImpFireSpread = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    RF_NoNullFireSpread = RandomForest.RandomForest(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    

    RF_IntSizeClass = RandomForest.RandomForest(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    # RF_ImpSizeClass = RandomForest.RandomForest(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    RF_ImpSizeClass = RandomForest.RandomForest(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    RF_NoNullSizeClass = RandomForest.RandomForest(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)
    

    # XGBoost - uses data with null values (dont use special test data here)
    XGB_FireSpread = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    XGB_SizeClass = XGBoost.XGBoost(dataRaw, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)

    #LightGBM
    LGBM_FireSpread = LightGBM.LightGBM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    LGBM_SizeClass = LightGBM.LightGBM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    LGBM_Int_FireSpread = LightGBM.LightGBM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    LGBM_Int_SizeClass = LightGBM.LightGBM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    LGBM_Null_FireSpread = LightGBM.LightGBM(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False)
    LGBM_Null_SizeClass = LightGBM.LightGBM(dataDropNull, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True)

    #SVM
    SVM_FireSpread = SVM.SVM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    SVM_SizeClass = SVM.SVM(dataImputed, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)
    SVM_Int_FireSpread = SVM.SVM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[0], predFeatCat = False, testData = testData)
    SVM_Int_SizeClass = SVM.SVM(dataInterpolated, data.quantFeat, data.catFeat, data.predictFeat[1], predFeatCat = True, testData = testData)


    #CREATES CSV OF ALL MODEL RESULTS
    result = pd.concat(
        [RF_ImpFireSpread.results, RF_ImpSizeClass.results,
         RF_IntFireSpread.results, RF_IntSizeClass.results,
         RF_NoNullFireSpread.results, RF_NoNullSizeClass.results,
         LGBM_FireSpread.results, LGBM_SizeClass.results, 
         LGBM_Int_FireSpread.results, LGBM_Int_SizeClass.results,
         LGBM_Null_FireSpread.results, LGBM_Null_SizeClass.results,
         XGB_FireSpread.results, XGB_SizeClass.results,
         SVM_FireSpread.results, SVM_SizeClass.results,
         SVM_Int_FireSpread.results, SVM_Int_SizeClass.results
        ], axis=1).transpose()
    result.to_csv("ModelResults/Results.csv", index=False)


if __name__ == "__main__":
    main()
