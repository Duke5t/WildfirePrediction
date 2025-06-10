from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split



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

        NUM_TREES_CAT = 400 # Number of trees in the forest - categorical predicition
        NUM_TREES_QUANT = 100 # Number of trees in the forest - quantitative predicition
        

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
            self.fireY = rawY


        #fireX is quant variables + OneHotEncoded cat variables
        self.fireX = pd.concat([self.df[quantFeat], self.dfCat], axis=1)


        #Create RandomForestClassifier
        #n_estimators = # of trees
        if predFeatCat:
            self.fireRF = RandomForestClassifier(n_estimators = NUM_TREES_CAT, oob_score = True)
        else:
            self.fireRF = RandomForestRegressor(n_estimators = NUM_TREES_QUANT, oob_score = True)
        
        #Split train/test if necessary (If there is no test data given)
        if testData is None:
            self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.fireX, self.fireY, test_size=0.2, stratify=self.fireY if predFeatCat else None)
        else: 
            self.trainX = self.fireX
            self.trainY = self.fireY
            self.testX = pd.concat([testData[quantFeat], self.encodeCat(testData, catFeat, fit=False)], axis=1)
            self.testY = testData[predFeat]
            if predFeatCat:
                self.testY = self.labelEncoder.transform(self.testY)



        #Train the classifier/regressor
        self.fireRF.fit(self.trainX, self.trainY) 



        print(f'Predicting {predFeat}.')


        # IF the feature we're predicting is categorical 
        if predFeatCat:
            print(f'Num classes: {len(self.labelEncoder.classes_)}')

            self.yHat = self.fireRF.predict(self.testX)
            
            print(f'Accuracy: {accuracy_score(self.testY, self.yHat):.4f}')

        # IF the feature we're predicting is NOT categorical 
        else:
            self.yHat = self.fireRF.predict(self.testX)

            mse = mean_squared_error(self.testY, self.yHat)
            r2 = r2_score(self.testY, self.yHat)
            print(f'MSE: {mse:.4f}')
            print(f'R^2: {r2:.4f}')
            # scores = cross_val_score(self.fireRF, self.fireX, self.fireY, cv=5, scoring='r2')
            # print(f"Cross-validated R² scores: {scores}")
            # print(f"Mean CV R²: {scores.mean():.4f}")



        print(f'Accuracy with training data (should be high): {self.fireRF.score(self.fireX, self.fireY):.4f}') #Prints accuracy when compared to training Data (should be really high)

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
        plt.show()

    def plotTestData(self):
        cm = confusion_matrix(self.testY, self.yHat)
        cmd = ConfusionMatrixDisplay(cm, display_labels=self.labelEncoder.classes_)
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Special Test Data)")
        plt.show()