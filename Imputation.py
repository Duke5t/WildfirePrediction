import pandas as pd
import numpy as np
import copy

#Imputation using mean / mode
#Uses mean values for quantitative features
#Uses mode values for categorical features

class Imputation:

    #args - original pandas dataframe, quantitative features, categorical features, actual feature to impute values
    #returns - data frame with null values replaced with imputed values. 
    def __init__(self, df, quantFeatures, catFeatures):

        self.df = df.copy() #includes only features being used
        self.showFeatureNAs(self.df)

        #runs imputation on each quantitative feature and updates dataframe NA cells with imputed values  

        print(f"Running Imputation on quantitative features..")
        self.quantFillMean(quantFeatures)

        #runs imputation on each categorical feature and updates dataframe NA cells with imputed values  
        print(f"Running Imputation on categorical features..")
        self.catFillMode(catFeatures)

        # Confirms final df has no null values remaining
        print(self.df.shape[0])
        self.showFeatureNAs(self.df)
        print(self.df.head(50))

    # Fill missing values in quantitative features with the mean
    def quantFillMean(self, cols):
        for col in cols:
            if self.df[col].isnull().any():
                mean = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean)

    #Takes Categorical data and fills NA values with mode
    def catFillMode(self, cols):
        # Fill missing values in each column with mode
        for col in cols:
            if self.df[col].isnull().any():
                mode = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode)
    # Shows total number of null/NaN values in each feature 
    def showFeatureNAs(self, df):
        print("NaNs in features:\n", df.isnull().sum()) 
