import Interpolation
import Imputation
import pandas as pd
import numpy as np
import os

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
        self.quantFeat = ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED"] # , "DISTANCE_FROM_WATER_SOURCE"
        self.catFeat = ["FUEL_TYPE", "TRUE_CAUSE", "DETECTION_AGENT", "DETECTION_AGENT_TYPE", "FIRE_POSITION_ON_SLOPE", "WEATHER_CONDITIONS_OVER_FIRE", "GENERAL_CAUSE"]
        self.predictFeat = ["FIRE_SPREAD_RATE", "SIZE_CLASS"]


    def createDataFrames(self):

        #Drops unused features
        selectedCols = self.quantFeat + self.catFeat + self.predictFeat
        self.df = self.df[selectedCols]
        
        #Removes firespread values <0 from DF
        self.df = self.df[self.df["FIRE_SPREAD_RATE"] >= 0]


        print(self.df.head(10))

        #Creates DF with no null values
        self.dfDropNull = self.df.copy().dropna().reset_index(drop=True) 


        #Creates a test set to be used later. 
        #Using 20% of the data but only if it has no null value
        ratio = self.df.shape[0]*self._testSplit/self.dfDropNull.shape[0]
        if(ratio > 1):
            raise Exception(f"Error: Not enough data without null values to populate Test DataFrame. Cannot make subset of {ratio: .2f}, x size.")
            
            self.dfTestData = self.dfDropNull ##Should I take the entire remaining data if test size cant be 20%???


        self.div = np.random.rand(len(self.dfDropNull)) < ratio
        #Sets special test subset containing complete data (no null, no interpolated, no imputed data)
        self.dfTestData = self.dfDropNull[self.div]
        self.dfTestData = self.dfTestData[self.dfTestData["FIRE_SPREAD_RATE"] >= 0]




        if not self.TEST:
            #Creates cleaned data by Interpolating with Lin and Log regression
            self.dfInterpolated = Interpolation.Interpolation(self.df, self.quantFeat, self.catFeat).df
            #Removes any data that is also contained in the test df
            self.dfInterpolated = pd.concat([self.dfInterpolated, self.dfTestData]).drop_duplicates(keep=False)

            #Creates cleaned data by Imputing mean and mode values.
            self.dfImputed = Imputation.Imputation(self.df, self.quantFeat, self.catFeat).df
            #Removes any data that is also contained in the test df
            self.dfImputed = pd.concat([self.dfImputed, self.dfTestData]).drop_duplicates(keep=False)





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
