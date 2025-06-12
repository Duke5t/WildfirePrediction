from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor, plot_tree
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



#TODO
#add evaluation


class XGBoost:

    # Constructor takes:
    #   df          = dataframe being used
    #   quantFeat   = list of quantitative features
    #   catFeat     = list of categorical features
    #   predFeat    = feature being predicted
    #   predFeatCat = boolean, if we're predicting a categorical variable
    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat = False):
        print(f"\n---XGBOOST---\n\":{predFeat}\"")

        ##Parameters##
        self.quantLearnRate = 1
        self.catLearnRate = 1
        self.quantNumBoostedTrees = 2
        self.catNumBoostedTrees = 2

        self.df = df.copy()
        self.labelEncoder = LabelEncoder()

        self.df = self.df.dropna(subset=[predFeat]) ##AVOIDS TRAINING/TESTING ON NULL VALUES

        for col in catFeat:
            self.df[col] = self.df[col].astype('category')


        selectedCols = quantFeat + catFeat
        fireX = self.df[selectedCols]
        fireY = self.df[predFeat]
        if predFeatCat:
            fireY = self.labelEncoder.fit_transform(fireY) #Encodes Y/predict feature


        X_train, X_test, y_train, y_test = train_test_split(fireX, fireY, test_size=0.2)


        #If we're training a categorical variable
        if predFeatCat:
            self.xgbCat(X_train, X_test, y_train, y_test)
            self.plotCat()
        else:
            self.xgbQuant(X_train, X_test, y_train, y_test)
            self.plotQuant()
                
        

    def xgbCat(self, X_train, X_test, y_train, y_test):
        #n_estimators = Number of gradient boosted trees. Equivalent to number of boosting rounds.
        #max_depth = tree depth/height
        #objective = learning model (log regression softmax = multi:softprob)
        #learning rate = scaler multiplier to default??? maybe???

        self.bst = XGBClassifier(n_estimators = self.catNumBoostedTrees, learning_rate = self.catLearnRate, 
                                objective = 'multi:softprob', enable_categorical = True, 
                                eval_metric = "mlogloss", tree_method = 'hist')

        # fit model
        eval_set = [(X_test, y_test)]
        self.bst.fit(X_train, y_train, eval_set = eval_set)

        preds = self.bst.predict(X_test)

        predsDecoded = self.labelEncoder.inverse_transform(preds)

        # return predsDecoded
        
    def xgbQuant(self, X_train, X_test, y_train, y_test):


        self.bst = XGBRegressor(n_estimators = self.quantNumBoostedTrees, learning_rate = self.quantLearnRate, 
                           objective = 'reg:squarederror', enable_categorical = True,
                           eval_metric = 'rmse', tree_method = 'hist')
        # fit model
        eval_set = [(X_test, y_test)]
        self.bst.fit(X_train, y_train, eval_set = eval_set)

        preds = self.bst.predict(X_test)

        # return preds


    def plotCat(self):
        results = self.bst.evals_result()
        plt.plot(results['validation_0']['mlogloss'], label='Loss')
        plt.title("XGBoost Log Loss Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Log Loss")
        plt.legend()
        plt.show()


    def plotQuant(self):
        pass