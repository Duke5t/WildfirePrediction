from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class XGBoost:


    def __init__(self, df, quantFeat, catFeat, predFeat):
        
        self.df = df.copy()
        selectedCols = quantFeat + catFeat

        for col in catFeat:
            self.df[col] = self.df[col].astype('category')
            
    
        fireX = self.df[selectedCols]
        fireY = self.df[predFeat]

        X_train, X_test, y_train, y_test = train_test_split(fireX, fireY, test_size=0.2)


        
        pass
