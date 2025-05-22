import MLRImputation
# import LRImputation
import pandas as pd
import os

class DataCleaning:

    def __init__(self, fileName):
        self._fileName = fileName
        self.df = self.createDataFrame()

        quantFeat = ["FIRE_SPREAD_RATE", "TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED", "ASSESSMENT_HECTARES"]
        catFeat = ["FIRE_TYPE", "FUEL_TYPE", "WIND_DIRECTION", "FIRE_POSITION_ON_SLOPE", "WEATHER_CONDITIONS_OVER_FIRE", "SIZE_CLASS"]

        #Drops unused features
        selectedCols = quantFeat + catFeat
        self.df = self.df[selectedCols]
               

        self.df = MLRImputation.MLRImputation(self.df, quantFeat, catFeat)
        # self.df = LRImputation.LRImputation(self.df, quantFeat, catFeat)
        
    #Takes data file (.xlsx) within same filepath/directory 
    #creates and returns pandas dataframe
    def createDataFrame(self):
        try:
            scriptDir = os.path.dirname(os.path.abspath(__file__))
            filePath = os.path.join(scriptDir, self._fileName)
            return pd.read_excel(filePath)
        except FileNotFoundError:
            print(f"Error: File '{self._fileName}' not found.")
            return None
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    
