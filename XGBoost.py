from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


#TODO
#- Apply SMOTE...


class XGBoost:

    # Constructor takes:
    #   df          = dataframe being used
    #   quantFeat   = list of quantitative features
    #   catFeat     = list of categorical features
    #   predFeat    = feature being predicted
    #   predFeatCat = boolean, if we're predicting a categorical variable
    #   testData    = DataFrame, Special predetermined set of test data to be used in model (not required)
    #                            column dimensions must match 'df' input. 
    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat = False, testData = None):
        print(f"\n---XGBOOST---\n\":{predFeat}\"")
        ##Toggle graphical results display##
        self._graphs = False #TOGGLE ME
        
        ##Parameters##
        self.quantLearnRate = 0.1
        self.catLearnRate = 1
        self.quantNumBoostedTrees = 100
        self.catNumBoostedTrees = 50

        self.quantFeat = quantFeat
        self.catFeat = catFeat
        self.selectedCols = quantFeat + catFeat
        self.predFeat = predFeat
        self.predFeatCat = predFeatCat
        self.testData = testData

        self.df = df.copy()
        self.labelEncoder = LabelEncoder()

        self.df = self.df.dropna(subset=[self.predFeat]) ##AVOIDS TRAINING/TESTING ON NULL VALUES

        for col in self.catFeat:
            self.df[col] = self.df[col].astype('category')
        
        fireX = self.df[self.selectedCols]
        fireY = self.df[self.predFeat]


        # If we're training a categorical variable and 
        # applying SMOTE 
        # (cant have nulls in our data so we use our 
        # special imputed or interpolated df 
        # along side special test data)
        if self.predFeatCat and self.testData is not None:
            # # Convert target labels to strings before any encoding
            # fireY = fireY.astype(str)
            # test_labels = self.testData[predFeat].astype(str)

            # fireX_encoded = pd.get_dummies(fireX, drop_first=True)
            # test_encoded = pd.get_dummies(self.testData[self.selectedCols], drop_first=True)

            # # Align feature columns between train and test sets
            # fireX_encoded, test_encoded = fireX_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

            # # Apply SMOTE on training features and string-converted labels
            # smote = SMOTE()
            # X_train_resampled, y_train_resampled = smote.fit_resample(fireX_encoded, fireY)

            # # Convert SMOTE output labels also to strings (just in case)
            # y_train_resampled = y_train_resampled.astype(str)

            # # Fit LabelEncoder on combined train and test labels (all strings)
            # combined_labels = pd.concat([pd.Series(y_train_resampled), test_labels])
            # self.labelEncoder = LabelEncoder()
            # self.labelEncoder.fit(combined_labels)

            # # Transform labels using the fitted encoder
            # y_train_encoded = self.labelEncoder.transform(y_train_resampled)
            # y_test_encoded = self.labelEncoder.transform(test_labels)

            # # Train with encoded labels and features
            # self.xgbCat(X_train_resampled, test_encoded, y_train_encoded, y_test_encoded)
            # self.plotCat()
            pass

        #If we're traomomg a categorical variable and we have nulls/dont want to use SMOTE (we use simple upsampling by duplicating examples from minority classes)
        elif self.predFeatCat: 
            fireY = self.labelEncoder.fit_transform(fireY)
            X_train, X_test, y_train, y_test = train_test_split(fireX, fireY, test_size=0.2)
            X_train, y_train = self.upsampleMinorityClasses(X_train, y_train)
            self.xgbCat(X_train, X_test, y_train, y_test)
            self.plotCat()

        elif self.testData is not None:
            pass

        else:
            ##applies log(1+x) to normalize fireSpread
            fireY_log = np.log1p(fireY)
            X_train, X_test, y_train_log, y_test_log = train_test_split(fireX, fireY_log, test_size=0.2)

            self.y_test_raw = self.df[self.predFeat].loc[y_test_log.index]  # to compare raw values
            self.xgbQuant(X_train, X_test, y_train_log, y_test_log)
            self.plotQuant()
        self.printFeatureImportance()

                
        

    def xgbCat(self, X_train, X_test, y_train, y_test):
        #n_estimators = Number of gradient boosted trees. Equivalent to number of boosting rounds.
        #max_depth = tree depth/height
        #objective = learning model (log regression softmax = multi:softprob)
        #learning rate = scaler multiplier to default??? maybe???

        self.bst = XGBClassifier(n_estimators=self.catNumBoostedTrees, learning_rate=self.catLearnRate, 
                                 objective='multi:softprob', enable_categorical=True, 
                                 eval_metric="mlogloss", early_stopping_rounds=10, tree_method='hist')


        ##ADDED early stopping to avoid overfitting
        self.bst.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = self.bst.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = self.computeF1(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")

        #Stores results for main class usage
        #index [model, MSE, RMSE, r^2, accuracy, f1score]
        self.results = pd.Series({
                    "Model": "XGBoost",
                    "MSE": None,
                    "RMSE": None,
                    "R2": None,
                    "Accuracy": acc,
                    "F1 Score": f1
                })

        
    def xgbQuant(self, X_train, X_test, y_train, y_test):

        X_train, y_train_log = self.sanitize_target(X_train, y_train, apply_log=True)
        X_test, y_test_log = self.sanitize_target(X_test, y_test, apply_log=True)


        self.bst = XGBRegressor(n_estimators=self.quantNumBoostedTrees, learning_rate=self.quantLearnRate, 
                                objective='reg:squarederror', enable_categorical=True,
                                eval_metric='rmse', early_stopping_rounds=10, tree_method='hist')


        ##ADDED early stopping to avoid overfitting
        self.bst.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], verbose=False)

        preds_log = self.bst.predict(X_test)
        preds = np.expm1(preds_log)  # inverse transform
        y_true = np.expm1(y_test_log)

        mse = mean_squared_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"R^2: {r2:.4f}")

        #Stores results for main class usage
        #index [model, MSE, RMSE, r^2, accuracy, f1score]
        self.results = pd.Series({
                "Model": "XGBoost",
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2,
                "Accuracy": None,
                "F1 Score": None
            })

    def plotCat(self):
        if self._graphs:
            results = self.bst.evals_result()
            plt.plot(results['validation_0']['mlogloss'], label='Loss')
            plt.title("XGBoost Log Loss Over Iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Log Loss")
            plt.legend()
            plt.show()


    def plotQuant(self):
        if self._graphs:
            results = self.bst.evals_result()
            plt.plot(results['validation_0']['rmse'], label='RMSE')
            plt.title("XGBoost RMSE Over Iterations")
            plt.xlabel("Iterations")
            plt.ylabel("RMSE")
            plt.legend()
            plt.show()


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


    def printFeatureImportance(self):
        importances = self.bst.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\n--- Feature Importances ---")
        for idx in sorted_idx:
            print(f"{self.selectedCols[idx]}: {importances[idx]:.4f}")

    def upsampleMinorityClasses(self, X, y):
        df = X.copy()
        df['label'] = y

        counts = df['label'].value_counts()
        max_count = counts.max()
        dfs = []

        for label in counts.index:
            df_label = df[df['label'] == label]
            df_upsampled = resample(df_label, replace=True, n_samples=max_count)
            dfs.append(df_upsampled)

        df_balanced = pd.concat(dfs)
        y_balanced = df_balanced['label'].values
        X_balanced = df_balanced.drop(columns='label')

        return X_balanced, y_balanced


    def sanitize_target(self, X, y_raw, apply_log=True, clip_negatives=True):
        #Clean and transform target values without dropping entire rows unnecessarily:
        #- Clip negative targets (if enabled)
        #- Apply log transform (optional)
        #- Drop rows where target becomes invalid
        #- Keep NaNs in X (since XGBoost handles them)

        y_clean = y_raw.copy()

        if clip_negatives:
            y_clean = y_clean.clip(lower=0)

        if apply_log:
            y_clean = y_clean.clip(lower=0)  # avoids log of negative numbers
            y_clean = np.log1p(y_clean)

        # Replace inf with null in y, then drop
        y_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_index = y_clean.dropna().index

        # Only filter X and y to remove rows with invalid target
        return X.loc[valid_index], y_clean.loc[valid_index]
    
