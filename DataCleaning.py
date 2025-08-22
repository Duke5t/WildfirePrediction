import Interpolation
import Imputation
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


class DataCleaning:

    def __init__(self, fileNames):
        print("\n---DATACLEANING---\n")

        self.TEST = False #Always default to false, will be toggled by main class

        self._testSplit = .2 #Default test split (20% original and complete(no null) data becomes exclusively for testing models)

        self.df = pd.DataFrame()


        # Read and combine all files into a single DataFrame
        if isinstance(fileNames, list):
            for fileName in fileNames:
                temp_df = self.importDataFrame(fileName)
                
                if temp_df is not None:
                    self.df = pd.concat([self.df, temp_df], ignore_index=True)
        else:
            temp_df = self.importDataFrame(fileNames)
            if temp_df is not None:
                self.df = temp_df


        #Dummy features:
        self.quantFeat = ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED"] 
        self.catFeat = ["FUEL_TYPE", "GENERAL_CAUSE", "DETECTION_AGENT_TYPE", 
                        "FIRE_POSITION_ON_SLOPE", "WEATHER_CONDITIONS_OVER_FIRE", 
                        "TRUE_CAUSE", "DETECTION_AGENT", "FIRE_START_DATE"] #"TRUE_CAUSE"
        self.predictFeat = ["FIRE_SPREAD_RATE", "SIZE_CLASS"]


    def createDataFrames(self):
        #Drops unused features
        selectedCols = self.quantFeat + self.catFeat + self.predictFeat
        self.df = self.df[selectedCols]
        print(self.df.shape)

        #Removes firespread values <0 from DF
        self.df = self.df[self.df["FIRE_SPREAD_RATE"] >= 0]

        #Trims Dates to Months
        self.df["FIRE_START_DATE"] = self.df["FIRE_START_DATE"].str[5:7]

        #Round Lat and long to 3 significant digits
        self.df["LATITUDE"] = round(self.df["LATITUDE"], 3)
        self.df["LONGITUDE"] = round(self.df["LONGITUDE"], 3)

        #Creates DF with no null values
        self.dfDropNull = self.df.copy().dropna().reset_index(drop=True) 

        #Creates a test set to be used later. 
        #Using 20% of the data but only if it has no null values
        ratio = self.df.shape[0]*self._testSplit/self.dfDropNull.shape[0]
        if(ratio > 1):
            raise Exception(f"Error: Not enough data without null values to populate Test DataFrame. Cannot make subset of {ratio: .2f}, x size.")
            
            self.dfTestData = self.dfDropNull ##Should I take the entire remaining data if test size cant be 20%???

        self.div = np.random.rand(len(self.dfDropNull)) < ratio
        #Sets special test subset containing complete data (no null, no interpolated, no imputed data)
        self.dfTestData = self.dfDropNull[self.div]
        self.dfTestData = self.dfTestData[self.dfTestData["FIRE_SPREAD_RATE"] >= 0]
        self.test_indices = self.dfTestData.index




        if not self.TEST:
            #Creates cleaned data by Interpolating with Lin and Log regression
            self.dfInterpolated = Interpolation.Interpolation(self.df, self.quantFeat, self.catFeat).df
            #Normalizes all quantitative Features
            self.dfInterpolated = self.normalize(self.dfInterpolated, self.quantFeat)
            #Removes any data that is also contained in the test df
            self.dfInterpolated = self.dfInterpolated[~self.dfInterpolated.index.isin(self.test_indices)]


            #Creates cleaned data by Imputing mean and mode values.
            self.dfImputed = Imputation.Imputation(self.df, self.quantFeat, self.catFeat).df
            #Normalizes all quantitative features
            self.dfImputed = self.normalize(self.dfImputed, self.quantFeat)
            #Removes any data that is also contained in the test df
            self.dfImputed = self.dfImputed[~self.dfImputed.index.isin(self.test_indices)]





    #Takes data file (.xlsx or .csv) within same filepath/directory 
    #creates and returns pandas dataframe
    def importDataFrame(self, fileName):
        try:
            filePath = os.path.abspath(fileName)
            
            #checks if csv or excel and reads to pd.DF accordingly
            if fileName.lower().endswith(".csv"):
                return pd.read_csv(filePath)
            elif fileName.lower().endswith(".xlsx"):
                return pd.read_excel(filePath)
            else:
                print("Error: Unsupported file type")
                return None
        except FileNotFoundError:
            print(f"Error: File '{fileName}' not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
        
    def setQuantitativeFeatures(self, features):
        self.quantFeat = features

    def setCategoricalFeatures(self, features):
        self.catFeat = features

    def setPredictionFeatures(self, features):
        self.predFeat = features

    # ratio = .2 by deafult (20% of original data with no null values becomes exclusively test data)
    def setTestSplit(self, ratio):
        self._testSplit = ratio

    def removeNegativeFireSpreadData(self):
        self.df = self.df[self.df["FIRE_SPREAD_RATE"] >= 0]

    def normalize(self, df, quantFeat):
        normalized_df = df.copy()

        for col in quantFeat:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            normalized_df[col] = (df[col] - mean) / std

        return normalized_df


    #FEATURE ENGINEERING:
    # TEMPERATURE × WIND_SPEED (fires spread faster in hot, windy conditions)
    # RELATIVE_HUMIDITY × FUEL_TYPE (moisture vs. burnable material)
    # Removes old features which were combined into engineered features
    def featureEngineering(self):

        #list of all dataframes
        dfs = [self.df, self.dfInterpolated, self.dfImputed, self.dfDropNull, self.dfTestData]

        for df in dfs:
            le = LabelEncoder()

            fuelType = le.fit_transform(df["FUEL_TYPE"])

            df["TEMP_AND_WIND"] = df["TEMPERATURE"] * df["WIND_SPEED"]
            df["FUEL_AND_HUMIDITY"] = df["RELATIVE_HUMIDITY"] * fuelType
            
            
            # Drop features used in engineering from all dfs
            df.drop(["TEMPERATURE", "WIND_SPEED", "RELATIVE_HUMIDITY", "FUEL_TYPE"], axis=1, inplace=True)

        
        # Change features list
        # ADD new features (engineered)
        if "TEMP_AND_WIND" not in self.quantFeat:
            self.quantFeat.append("TEMP_AND_WIND")
        if "FUEL_AND_HUMIDITY" not in self.quantFeat:
            self.quantFeat.append("FUEL_AND_HUMIDITY")

        # REMOVE old features used
        for feat in ["TEMPERATURE", "WIND_SPEED", "RELATIVE_HUMIDITY"]:
            if feat in self.quantFeat:
                self.quantFeat.remove(feat)

        if "FUEL_TYPE" in self.catFeat:
            self.catFeat.remove("FUEL_TYPE")

        print("FEATURE ENGINEERING PRINTOUT:")
        print(self.quantFeat)
        print(self.catFeat)


    