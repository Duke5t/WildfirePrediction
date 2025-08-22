from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import Evaluation


#TODO
# add option for test data
class SVM():

    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat=False, testData=None):
        print(f"\n---SUPPORT VECTOR MACHINE---\n\"{predFeat}\"")

        self.quantFeat = quantFeat
        self.catFeat = catFeat
        self.selectedCols = quantFeat + catFeat
        self.predFeat = predFeat
        self.predFeatCat = predFeatCat
        self.testData = testData
        self.df = df.dropna(subset=[predFeat]).copy()
        self.smote = SMOTE()
        self.labelEncoder = LabelEncoder()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


        trainX = self.df[self.selectedCols]
        trainY = self.df[self.predFeat]
        testX = testData[self.selectedCols]
        testY = testData[self.predFeat]

        trainX, testX = self.standardizeAndEncode(trainX, testX)

        if self.predFeatCat:
            #Encode Target
            trainY = self.labelEncoder.fit_transform(trainY)
            testY = self.labelEncoder.fit_transform(testY)

            #Oversample minority class using SMOTE
            trainX, trainY = self.smote.fit_resample(trainX, trainY)

            #Run Cat SVM Model
            self.svmCat(trainX, testX, trainY, testY)

        else:
            # Normalize target 'FIRE SPREAD'
            trainY_log = np.log1p(trainY)
            testY_log = np.log1p(testY)
            
            self.svmQuant(trainX, testX, trainY_log, testY_log)


    def svmCat(self, X_train, X_test, y_train, y_test):
        self.model = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale' , random_state = 1)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)


        # Calculates Evaluation metrics
        self.eval = Evaluation.Evaluation(preds, y_test, "SVM", categorical=True)
        self.results = self.eval.results


    def svmQuant(self, X_train, X_test, y_train_log, y_test_log):
        self.model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        self.model.fit(X_train, y_train_log)
        preds_log = self.model.predict(X_test)

        preds = np.expm1(preds_log)
        y_test = np.expm1(y_test_log)

        # Calculates Evaluation metrics
        self.eval = Evaluation.Evaluation(preds, y_test, "SVM", categorical=False)
        self.results = self.eval.results


    def computeF1(self, y_true, y_pred):
        # Ensures all labels are represented
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        precs = []
        recalls = []

        for label in labels:
            tp = np.sum((y_pred == label) & (y_true == label))
            fp = np.sum((y_pred == label) & (y_true != label))
            fn = np.sum((y_pred != label) & (y_true == label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precs.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return np.mean(precs), np.mean(recall), np.mean(f1s)

    def upsampleMinorityClasses(self, X, y):
        df = X.copy()
        df['label'] = y
        counts = df['label'].value_counts()
        max_count = counts.max()
        dfs = [resample(df[df['label'] == label], replace=True, n_samples=max_count)
               for label in counts.index]
        df_balanced = pd.concat(dfs)
        y_balanced = df_balanced['label'].values
        X_balanced = df_balanced.drop(columns='label')
        return X_balanced, y_balanced

    def standardizeAndEncode(self, trainX, testX):
        trainX = trainX.copy().reset_index(drop=True)
        testX = testX.copy().reset_index(drop=True)

        combinedX = pd.concat([trainX, testX], axis=0).reset_index(drop=True)

        # One-hot encode categorical columns
        if self.catFeat:
            self.ohe.fit(combinedX[self.catFeat])
            combinedCatEncoded = pd.DataFrame(
                self.ohe.transform(combinedX[self.catFeat]),
                columns=self.ohe.get_feature_names_out(self.catFeat),
            )
        else:
            combinedCatEncoded = pd.DataFrame(index=combinedX.index)

        # Standardize quantitative features
        scaler = StandardScaler()
        combinedQuantScaled = pd.DataFrame(
            scaler.fit_transform(combinedX[self.quantFeat]),
            columns=self.quantFeat
        )

        # Merge encoded and scaled data
        combinedProcessed = pd.concat([combinedQuantScaled, combinedCatEncoded], axis=1)

        # Split back into train and test
        trainXProcessed = combinedProcessed.iloc[:len(trainX)].reset_index(drop=True)
        testXProcessed = combinedProcessed.iloc[len(trainX):].reset_index(drop=True)

        return trainXProcessed, testXProcessed
