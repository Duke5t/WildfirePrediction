from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE



# Bagging - Bootstrap aggregation : manufacturing different data 
# Bagging - Manufactures data with same size as original data but with potential duplicates/overrepresentation (Bags)
# Bagging works with random sampling with replacement
# Bagging - Compliment of Bag 1 and original data is called "Out Of Bag Data" (data that is in Original and not in Bag1)

# Feature Randomization - Each bag is assigned a subset of features (formula: num features in subset = sqrt(total features))


#Random Forest Takes the interpolated data.
#Tests against specific test data and out of bag data
class RandomForest:
    # How do I choose how many estimators for random forest (n_estimators = how many trees)?
    # OOB score uses OOB data as test data



    # Constructor:
    #   df          = dataframe being used
    #   quantFeat   = list of quantitative features
    #   catFeat     = list of categorical features
    #   predFeat    = feature being predicted
    #   predFeatCat = boolean, if we're predicting a categorical variable
    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat = False, testData = None):
        print("\n---RANDOM FOREST---\n")

        NUM_TREES_CAT = 900 # Number of trees in the forest - categorical predicition
        NUM_TREES_QUANT = 300 # Number of trees in the forest - quantitative predicition
        

        self.df = df.copy()
        self.df = self.df.dropna(subset=[predFeat]) ##AVOIDS TRAINING/TESTING ON NULL VALUES


        self.dfCat = self.encodeCat(self.df, catFeat)

        #Specific only to the feature we're predicting
        self.labelEncoder = LabelEncoder()

        rawY = self.df[predFeat]

        # Combine test labels if needed for encoding
        if predFeatCat and testData is not None:
            allLabels = pd.concat([rawY, testData[predFeat]], axis=0)            
            self.labelEncoder.fit(allLabels)
            self.fireY = self.labelEncoder.transform(rawY)
        elif predFeatCat:
            self.fireY = self.labelEncoder.fit_transform(rawY)
        else:
            # Apply log transform for skewed data
            self.fireY = np.log1p(rawY)

        #fireX is quant variables + OneHotEncoded cat variables
        self.fireX = pd.concat([self.df[quantFeat], self.dfCat], axis=1)


        #Create RandomForestClassifier
        #n_estimators = # of trees
        if predFeatCat:
            self.fireRF = RandomForestClassifier(n_estimators = NUM_TREES_CAT, max_depth=20, oob_score = True)
        else:
            self.fireRF = RandomForestRegressor(n_estimators = NUM_TREES_QUANT, max_depth=20, oob_score = True)
        
        #Split train/test if necessary (If there is no test data given)
        if testData is None:
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(
                self.fireX, self.fireY, test_size=0.2, stratify=self.fireY if predFeatCat else None)
            
            if predFeatCat:
                # Apply SMOTE for categorical underrepresented classes
                smote = SMOTE()
                self.trainX, self.trainY = smote.fit_resample(self.trainX, self.trainY)
        else: 
            self.trainX = self.fireX
            self.trainY = self.fireY
            self.testX = pd.concat([testData[quantFeat], self.encodeCat(testData, catFeat, fit=False)], axis=1)
            self.testY = testData[predFeat]
            if predFeatCat:
                self.testY = self.labelEncoder.transform(self.testY)
            else:
                #Log transform test
                self.testY = np.log1p(self.testY)


        #Train the classifier/regressor
        self.fireRF.fit(self.trainX, self.trainY) 



        print(f'Predicting {predFeat}.')
        if testData is not None:
            print("(With Special Test Data)")


        # IF the feature we're predicting is categorical 
        if predFeatCat:
            print(f'Num classes: {len(self.labelEncoder.classes_)}')

            self.yHat = self.fireRF.predict(self.testX)
            

            acc = accuracy_score(self.testY, self.yHat)
            f1 = self.computeF1(self.testY, self.yHat)
            print(f'Accuracy: {acc:.4f} With {NUM_TREES_CAT} Trees')
            print(f'F1: {f1:.4f}')

            #Stores results for main class usage
            #index [model, MSE, RMSE, r^2, accuracy, f1score]
            self.results = pd.Series({
                    "Model": "Random Forest",
                    "MSE": None,
                    "RMSE": None,
                    "R2": None,
                    "Accuracy": acc,
                    "F1 Score": f1
                })

        # IF the feature we're predicting is NOT categorical 
        else:
            self.yHat = self.fireRF.predict(self.testX)

            # Inverse log transform predictions and targets
            yHat_original = np.expm1(self.yHat)
            testY_original = np.expm1(self.testY)

            mse = mean_squared_error(testY_original, yHat_original)
            r2 = r2_score(testY_original, yHat_original)
            print(f'MSE: {mse:.4f} ({NUM_TREES_QUANT} Trees)')
            print(f'RMSE: {np.sqrt(mse):.4f} ({NUM_TREES_QUANT} Trees)')
            print(f'R^2: {r2:.4f} ({NUM_TREES_QUANT} Trees)')

            #Stores results for main class usage
            #index [model, MSE, RMSE, r^2, accuracy, f1score]
            self.results = pd.Series({
                    "Model": "Random Forest",
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "R2": r2,
                    "Accuracy": None,
                    "F1 Score": None
                })

        if predFeatCat:
            self.plotOOB(testData)
            self.plotTestData()
            print(f'OOB data score: {self.fireRF.oob_score_:.4f}') #Prints accuracy when compared to OOB Data (Out Of Bag)
        else:
            print(f'OOB R^2 score: {self.fireRF.oob_score_:.4f}')
            

    def encodeCat(self, df, catFeat, fit = True):
        if fit:
            self.oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Set up One Hot Encoder
            encodedArray = self.oneHotEncoder.fit_transform(df[catFeat])
        else:
            encodedArray = self.oneHotEncoder.transform(df[catFeat])
        
        featureNames = self.oneHotEncoder.get_feature_names_out(catFeat)

        return pd.DataFrame(encodedArray, columns=featureNames, index=df[catFeat].index)
    
    def plotOOB(self, testData):
        oobPreds = pd.DataFrame(self.fireRF.oob_decision_function_, columns = self.labelEncoder.classes_)
        oobPreds['Label'] = oobPreds.values.argmax(axis=1)

        cm = confusion_matrix(self.trainY, oobPreds['Label'])
        cmd = ConfusionMatrixDisplay(cm, display_labels = self.labelEncoder.classes_)
        cmd.plot(cmap=plt.cm.Greens)
        plt.title("Confusion Matrix: OOB DATA")
        # plt.show()

    def plotTestData(self):
        cm = confusion_matrix(self.testY, self.yHat)
        cmd = ConfusionMatrixDisplay(cm, display_labels=self.labelEncoder.classes_)
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        # plt.show()

    def computeF1(self, y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for label in labels:
            tp = np.sum((y_pred == label) & (y_true == label))
            fp = np.sum((y_pred == label) & (y_true != label))
            fn = np.sum((y_pred != label) & (y_true == label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1s.append(f1)
        return np.mean(f1s)  # macro average