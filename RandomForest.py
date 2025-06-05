from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt


# Bagging - Bootstrap aggregation : manufacturing different data 
# Bagging - Manufactures data with same size as original data but with potential duplicates/overrepresentation (Bags)
# Bagging works with random sampling with replacement
# Bagging - Compliment of Bag 1 and original data is called "Out Of Bag Data" (data that is in Original and not in Bag1)

# Feature Randomization - Each bag is assigned a subset of features (formula: num features in subset = sqrt(total features))


# 

class RandomForest:


        # How do I choose how many estimators for random forest (n_estimators = how many trees)?
        # OOB score uses OOB data as test data

    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat = False, testData = None):
        
        self.df = df.copy()

        self.dfCat = self.encodeCat(df, catFeat)

        #Specific only to the feature we're predicting
        self.labelEncoder = LabelEncoder()

        #fireY is predicted variable encoded (fireY = labelEncoder.fit_transform( [##single predictFeat] ))
        self.fireY = self.labelEncoder.fit_transform(df[predFeat])

        #fireX is quant variables + OneHotEncoded cat variables
        self.fireX = pd.concat([df[quantFeat], self.dfCat], axis=1)


        #Create RandomForestClassifier
        #n_estimators = # of trees
        self.fireRF = RandomForestClassifier(n_estimators = 400, oob_score = True)


        #Train the classifier
        self.fireRF.fit(self.fireX, self.fireY) 


        print(f'Predicting {predFeat}. Num classes: {len(self.labelEncoder.classes_)}')
        print(f'Accuracy with OOB data: {self.fireRF.oob_score_:.4f}') #Prints accuracy when compared to OOB Data (Out Of Bag)

        if testData is not None:
            self.testX = pd.concat([testData[quantFeat], self.encodeCat(testData,catFeat, fit=False)], axis=1)
            self.testY = self.labelEncoder.transform(testData[predFeat])
            self.yHat = self.fireRF.predict(self.testX)
            
            print(f'Accuracy with special test data (no interpolation or imputation): {accuracy_score(self.testY, self.yHat):.4f}')

        print(f'Accuracy with training data (should be high): {self.fireRF.score(self.fireX, self.fireY):.4f}') #Prints accuracy when compared to training Data (should be really high)

        if predFeatCat:
            self.plotOOB();
            self.plotTestData()

            

    def encodeCat(self, df, catFeat, fit = True):
        if fit:
            self.oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Set up One Hot Encoder
            encodedArray = self.oneHotEncoder.fit_transform(df[catFeat])
        else:
            encodedArray = self.oneHotEncoder.transform(df[catFeat])
        
        featureNames = self.oneHotEncoder.get_feature_names_out(catFeat)

        return pd.DataFrame(encodedArray, columns=featureNames, index=df[catFeat].index)
    
    def plotOOB(self):
        oobPreds = pd.DataFrame(self.fireRF.oob_decision_function_, columns = self.labelEncoder.classes_) #References label encoder for y data
        oobPreds['Label'] = oobPreds.values.argmax(axis=1)


        #plots confusion matrix (shows TP, FP, TN, FN) (helpful for visualization) (helpful for F1?)
        cm = confusion_matrix(self.fireY, oobPreds['Label'])
        cmd = ConfusionMatrixDisplay(cm, display_labels = self.labelEncoder.classes_) #References label encoder for y data
        cmd.plot(cmap=plt.cm.Greens)
        plt.title("Confusion Matrix: OOB DATA")
        plt.show()

    def plotTestData(self):
        cm = confusion_matrix(self.testY, self.yHat)
        cmd = ConfusionMatrixDisplay(cm, display_labels=self.labelEncoder.classes_)
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Special Test Data)")
        plt.show()

