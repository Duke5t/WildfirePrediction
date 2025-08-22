from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import pandas as pd


# Calculates:
#    Accuracy : acc
#    Precision : prec
#    Recall : recall
#    F1 (Macro) : f1
#    Mean Squared Error : mse
#    Root Mean Squared Error : rmse
#    Correlation : r2


class Evaluation:
    
    # Takes vectors of predictions and actuals from test set.
    # Computes ands stores evaluation methods and results
    def __init__(self, prediction, actual, model, categorical = False):
        self.prediction = prediction
        self.actual = actual
        self.categorical = categorical
        
        self.model = model
        self._metricsInput()
        self._formatResults()


    def _metricsInput(self):
        if self.categorical:
            self.acc = accuracy_score(self.actual, self.prediction)
            self.prec, self.recall, self.f1  = self._computeF1()
        else:
            self.mse = mean_squared_error(self.actual, self.prediction)
            self.rmse = np.sqrt(self.mse)
            self.r2 = r2_score(self.actual, self.prediction)


    
    # Formats and stores results for main class usage
    def _formatResults(self):
        if self.categorical:
            self.results = pd.Series({
                "Model": self.model,
                "MSE": None,
                "RMSE": None,
                "R2": None,
                "Accuracy": f"{self.acc:.3f}",
                "Precision (Avg)": f"{self.prec:.3f}",
                "Recall (Avg)": f"{self.recall:.3f}",
                "F1 Score (Macro)": f"{self.f1:.3f}"
            })
        else:
            self.results = pd.Series({
                "Model": self.model,
                "MSE": f"{self.mse:.3f}",
                "RMSE": f"{self.rmse:.3f}",
                "R2": f"{self.r2:.3f}",
                "Accuracy": None,
                "F1 Score (Macro)": None
            })
    
    # Computes Macro F1
    def _computeF1(self):
        # Ensures all labels are represented
        labels = np.unique(np.concatenate([self.actual, self.prediction]))
        f1s = []
        precs = []
        recalls = []

        for label in labels:
            tp = np.sum((self.prediction == label) & (self.actual == label))
            fp = np.sum((self.prediction == label) & (self.actual != label))
            fn = np.sum((self.prediction != label) & (self.actual == label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            precs.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return np.mean(precs), np.mean(recalls), np.mean(f1s)