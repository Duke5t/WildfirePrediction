import Interpolation
import Imputation
import pandas as pd
import os

class DataCleaning:

    def __init__(self, fileNames):
        self.df = pd.DataFrame()

        # Read and combine all files into a single DataFrame
        if isinstance(fileNames, list):
            for fileName in fileNames:
                temp_df = self.createDataFrame(fileName)
                print(temp_df.shape[0])
                print(temp_df.columns.values)
                if temp_df is not None:
                    self.df = pd.concat([self.df, temp_df], ignore_index=True)
        else:
            temp_df = self.createDataFrame(fileNames)
            if temp_df is not None:
                self.df = temp_df

        self.quantFeat = ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED"]
        self.catFeat = ["FUEL_TYPE", "WIND_DIRECTION", "FIRE_POSITION_ON_SLOPE", "WEATHER_CONDITIONS_OVER_FIRE", "GENERAL_CAUSE"]#, "SIZE_CLASS"] #FIRE DATE

        self.predictFeat = ["FIRE_SPREAD_RATE", "SIZE_CLASS"] # 

        #Drops unused features
        selectedCols = self.quantFeat + self.catFeat + self.predictFeat
        self.df = self.df[selectedCols]

        print(self.df.head(50))
               


        #Creates cleaned data by Interpolating with Lin and Log regression
        self.dfInterpolated = Interpolation.Interpolation(self.df, self.quantFeat, self.catFeat).df
        
        #Creates cleaned data by dropping null values
        self.dfDropNull = self.df.copy().dropna().reset_index(drop=True) 
        
        #Creates cleaned data by Imputing mean and mode values.
        self.dfImputed = Imputation.Imputation(self.df, self.quantFeat, self.catFeat)



    #Takes data file (.xlsx or .csv) within same filepath/directory 
    #creates and returns pandas dataframe
    def createDataFrame(self, fileName):
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