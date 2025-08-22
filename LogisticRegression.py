import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



class LogisticRegression:

    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat = False, testData = None):
        self.df = df.copy()
        self.quantFeat = quantFeat
        self.catFeat = catFeat
        self.predFeat = predFeat
        self.predFeatCat = predFeatCat
        self.testData = testData

        
        self.df = df.dropna(subset=[predFeat]).copy()
        self.smote = SMOTE()
        self.labelEncoder = LabelEncoder()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


        trainX = self.df[self.selectedCols]
        trainY= self.df[self.predFeat]

        trainX = self.standardizeAndEncode(trainX)
        testX = self.standardizeAndEncode(testX)

        if self.predFeatCat:
            #Encode Target
            trainY = self.labelEncoder.fit_transform(trainY)
            testY = self.labelEncoder.fit_transform(testY)

            #Oversample minority class using SMOTE
            trainX, trainY = self.smote.fit_resample(trainX, trainY)

            #Run Cat SVM Model
            self.logRegCat(trainX, testX, trainY, testY)

        else:
            # Normalize target 'FIRE SPREAD'
            trainY_log = np.log1p(trainY)
            testY_log = np.log1p(testY)
            
            self.logRegQuant(trainX, testX, trainY_log, testY_log)
    
    def standardizeAndEncode(self, fireX):
        # One-hot encode categorical columns
        if self.catFeat:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = ohe.fit_transform(fireX[self.catFeat])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(self.catFeat), index=fireX.index)
            fireX = pd.concat([fireX[self.quantFeat], encoded_df], axis=1)

        # Standardize features
        scaler = StandardScaler()
        fireX[self.quantFeat] = scaler.fit_transform(fireX[self.quantFeat])

        return fireX