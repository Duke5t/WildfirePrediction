import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

class LightGBM:
    def __init__(self, df, quantFeat, catFeat, predFeat, predFeatCat=False, testData=None):
        self.quantFeat = quantFeat
        self.catFeat = catFeat
        self.selectedCols = quantFeat + catFeat
        self.predFeat = predFeat
        self.predFeatCat = predFeatCat
        self.df = df.copy().dropna(subset=[predFeat])
        self.testData = testData
        self.encoder = LabelEncoder()

        # One-hot encode training data
        X, y = self._prepare_data(self.df)

        if testData is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        else:
            X_train, y_train = X, y
            X_test, y_test = self._prepare_test_data(self.testData)

        self._train(X_train, y_train)

        if y_test is not None:
            self._evaluate(X_test, y_test)
        else:
            self.predictions = self.model.predict(X_test)

    def _prepare_data(self, df):
        df = df.copy()
        if self.predFeatCat:
            y = self.encoder.fit_transform(df[self.predFeat])
        else:
            y = df[self.predFeat]

        X_cat = pd.get_dummies(df[self.catFeat], drop_first=True)
        X = pd.concat([df[self.quantFeat], X_cat], axis=1)
        self.train_columns = X.columns  # store columns for later alignment
        return X, y

    def _prepare_test_data(self, df):
        df = df.copy()
        if self.predFeatCat:
            y = self.encoder.transform(df[self.predFeat])
        else:
            y = df[self.predFeat]

        X_cat = pd.get_dummies(df[self.catFeat], drop_first=True)
        X_cat = X_cat.reindex(columns=[col for col in self.train_columns if col not in self.quantFeat], fill_value=0)
        X_quant = df[self.quantFeat]
        X = pd.concat([X_quant, X_cat], axis=1)
        X = X.reindex(columns=self.train_columns, fill_value=0)
        return X, y

    def _train(self, X_train, y_train):
        if self.predFeatCat:
            self.model = lgb.LGBMClassifier()
        else:
            self.model = lgb.LGBMRegressor()
        self.model.fit(X_train, y_train)

    def _evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(f"\nEvaluating prediction for '{self.predFeat}'")
        if self.predFeatCat:
            acc = accuracy_score(y_test, y_pred)
            f1 = self.computeF1(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4}")

            #Stores results for main class usage
            #index [model, MSE, RMSE, r^2, accuracy, f1score]
            self.results = pd.Series({
                    "Model": "LightGBM",
                    "MSE": None,
                    "RMSE": None,
                    "R2": None,
                    "Accuracy": acc,
                    "F1 Score": f1
                })

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {np.sqrt(mse):.4f}")
            print(f"R^2 Score: {r2:.4f}")
            
            #Stores results for main class usage
            #index [model, MSE, RMSE, r^2, accuracy, f1score]
            self.results = pd.Series({
                    "Model": "LightGBM",
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "R2": r2,
                    "Accuracy": None,
                    "F1 Score": None
                })

    def predict(self, df):
        df = df.copy()
        X_cat = pd.get_dummies(df[self.catFeat], drop_first=True)
        X_cat = X_cat.reindex(columns=[col for col in self.train_columns if col not in self.quantFeat], fill_value=0)
        X_quant = df[self.quantFeat]
        X = pd.concat([X_quant, X_cat], axis=1)
        X = X.reindex(columns=self.train_columns, fill_value=0)

        preds = self.model.predict(X)
        if self.predFeatCat:
            preds = self.encoder.inverse_transform(preds)
        return preds

    def feature_importances(self):
        print("\nFeature Importances:")
        for col, score in sorted(zip(self.train_columns, self.model.feature_importances_), key=lambda x: -x[1]):
            print(f"  {col}: {score}")


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