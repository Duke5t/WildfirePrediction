from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import Evaluation
from imblearn.over_sampling import SMOTE



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

        ##Toggle hyperparameter tuning##
        self._tuneParams = False #TOGGLE ME

        self.quantFeat = quantFeat
        self.catFeat = catFeat
        self.selectedCols = quantFeat + catFeat
        self.predFeat = predFeat
        self.predFeatCat = predFeatCat
        self.testData = testData

        self.df = df.copy()
        self.labelEncoder = LabelEncoder()

        self.initializeHyperparams()

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

        #If we're training a categorical variable and we have nulls/dont want to use SMOTE (we use simple upsampling by duplicating examples from minority classes)
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

        if self._tuneParams:
            self.bst = XGBClassifier(enable_categorical=True, tree_method='hist', use_label_encoder=False, random_state = 1)
            param_dist = {**self.xgb_common_params, **self.xgb_class_params}
    
            search = RandomizedSearchCV(
                estimator=self.bst,
                param_distributions=param_dist,
                n_iter=30,  # Adjust as needed
                scoring='f1_macro',
                cv=3,
                verbose=0,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            self.bst = search.best_estimator_
            print("Best XGBoost Hyperparams:")
            print(search.best_estimator_)
        
        #If not tuning hyperparameters
        else:
            self.bst = XGBClassifier(n_estimators = 500, learning_rate = 0.2, objective='multi:softprob', 
                    enable_categorical=True, eval_metric="mlogloss", early_stopping_rounds=10, tree_method='hist', 
                    colsample_bytree=1.0, gamma=0.05, max_depth=7, min_child_weight=1, use_label_encoder=False, random_state = 1
                )
            
            # Early stopping to avoid overfitting
            self.bst.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        
        preds = self.bst.predict(X_test)

        # Plot Confusion Matrix
        if self._graphs:
            self.plotTestData(preds, y_test)
        
        # Calculates Evaluation metrics
        self.eval = Evaluation.Evaluation(preds, y_test, "XGBoost", categorical=True)
        self.results = self.eval.results

    
    def xgbQuant(self, X_train, X_test, y_train, y_test):

        X_train, y_train_log = self.sanitize_target(X_train, y_train, apply_log=True)
        X_test, y_test_log = self.sanitize_target(X_test, y_test, apply_log=True)


        #If hyperparameter tuning
        if self._tuneParams:
            self.bst = XGBRegressor(enable_categorical=True, tree_method='hist', random_state = 1)
            param_dist = {**self.xgb_common_params, **self.xgb_reg_params}

            search = RandomizedSearchCV(
                estimator=self.bst,
                param_distributions=param_dist,
                n_iter=30,
                scoring='r2',
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            search.fit(X_train, y_train_log)
            self.bst = search.best_estimator_
            print("\nBest Hyperparameters (Regression):")
            print(search.best_params_)

        #If not hyperparameter tuning
        else:
            self.bst = XGBRegressor(n_estimators = 500, learning_rate = 0.05, 
                        objective = 'reg:squarederror', enable_categorical=True, tree_method='hist', eval_metric = 'rmse', 
                        min_child_weight = 5, max_depth = 10,  gamma = 0.2, subsample = 0.8, reg_lambda = 1, reg_alpha = 1, 
                        colsample_bytree = 0.6, early_stopping_rounds=10, random_state = 1)


            ##Early stopping to avoid overfitting
            self.bst.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], verbose=False)

        preds_log = self.bst.predict(X_test)
        preds = np.expm1(preds_log)  # inverse transform
        y_test = np.expm1(y_test_log)
        
        #  Calculates Evaluation metrics
        self.eval = Evaluation.Evaluation(preds, y_test, "XGBoost", categorical=False)
        self.results = self.eval.results


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

    def plotTestData(self, preds, y_test):
        cm = confusion_matrix(y_test, preds)
        cmd = ConfusionMatrixDisplay(cm, display_labels=self.labelEncoder.classes_)
        cmd.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()


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
    

    def initializeHyperparams(self):
        self.xgb_common_params = {
            'n_estimators': [100, 200, 300, 400, 500, 600],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10, 12],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0.1, 0.5, 1, 1.5]
        }

        self.xgb_class_params = {
            'objective': ['multi:softprob'],
            'eval_metric': ['mlogloss']
        }

        self.xgb_reg_params = {
            'objective': ['reg:squarederror'],
            'eval_metric': ['rmse']
        }