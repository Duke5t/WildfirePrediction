# The purpose of this class will be to use multiple linear regression 
# to predict the fuel type of the fire (type of trees/vegetation) based on the following factors
# Quantitative Data:
# - fire_spread_rate
# - temperature
# - relative_humidity
# - wind_speed
# - assessment_hectares
# 
# Categorical Data:
# - fire_type
# - fire_position_on_slope
# - weather_conditions_over_fire
# - wind_direction
# - size_class


#TODO 
# set / choose initial values for gradient descent
# Choose learning rate alpha
# Choose convergence value epsilon

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
import copy

#Multiple Linear Regression Imputation
class Imputation:

    _linRegAlpha = 0.01 ##Learning Rate for Linear Regression
    _linRegEpsilon = 0.001 ##Convergence Rate for Linear Regression
    _logRegAlpha = 0.01 ##Learning Rate for Logisitc Regression
    _logRegEpsilon = 0.001 ##Convergence Rate for Logistic Regression


    #args - original pandas dataframe, quantitative features, categorical features, actual feature to imputate vales
    #returns - data frame with null values replaced with imputated values. 
    def __init__(self, df, quantFeatures, catFeatures):

        self.df = df.copy() #includes only features being used
        print(self.df.shape[0])
        self.showFeatureNAs(self.df)


        #runs imputation on each quantitative feature and updates dataframe NA cells with imputated values  
        for actual in quantFeatures:
            rQuantFeatures = [feature for feature in quantFeatures if feature != actual]
            
            print(f"Running Imputation on :{actual}")
            self.prepData(df, rQuantFeatures, catFeatures, actual)
            self.runQuantImputation(rQuantFeatures, catFeatures, actual)

        #runs imputation on each categorical feature and updates dataframe NA cells with imputated values  
        for actual in catFeatures:
            rCatFeatures = [feature for feature in catFeatures if feature != actual]

            print(f"Running Imputation on :{actual}")
            self.prepData(df, quantFeatures, rCatFeatures, actual, True)
            self.runCatImputation(quantFeatures, rCatFeatures, actual)

        print(self.df.shape[0])
        self.showFeatureNAs(self.df)


    # Drops rows with any NA values
    # Normalizes data
    # Sets test and training subsets (80/20)
    def prepData(self, df, quantFeatures, catFeatures, actual, isActualCategorical = False):
        # Shows total number of null/NaN values in each feature 
        # self.showFeatureNAs(df)

        #Drops rows which contain NA values in any selected column
        self.dfNoNull = df.copy().dropna().reset_index(drop=True)
        print(f"# of Rows in DF without null vals: {self.dfNoNull.shape[0]}")
        #Creates copies of df for quantitative, catergorical features and the intended/actual feature to imputate
        self.dfQuant = self.dfNoNull[quantFeatures].copy()
        self.dfCat = self.dfNoNull[catFeatures].copy()
        self.dfActual = self.dfNoNull[actual].copy()

        self.normalizedDF = self.normalizeData(self.dfQuant, self.dfCat, quantFeatures, catFeatures)

        ##Divide 80/20 train/test data
        div = np.random.rand(len(self.dfNoNull)) < 0.8
        self.xTrain = self.normalizedDF[div].values
        self.xTest = self.normalizedDF[~div].values
        if isActualCategorical:
            self.yTrain = self.dfActual[div].astype(str).str.strip().values
            self.yTest = self.dfActual[~div].astype(str).str.strip().values
        else:
            self.yTrain = self.dfActual[div].values.flatten()
            self.yTest = self.dfActual[~div].values.flatten()
            
    
    def runQuantImputation(self, quantFeatures, catFeatures, actual):
        # Set initial values for gradient descent
        w0 = np.zeros(self.xTrain.shape[1])
        b0 = 0
        alpha = self._linRegAlpha #Learning rate

        weightVector, intercept, history = self.gradientDescent(self.xTrain, self.yTrain, w0, b0, alpha)
        print(f"intercept, weights found by gradient descent: {intercept:0.2f}, {weightVector} ")
        # print(f"history: {history}")

        yPred = self.predict(self.xTest, weightVector, intercept)
        mse = np.mean((yPred - self.yTest) ** 2)
        print(f"Test MSE: {mse:.2f}")

        self.updateDF(actual, weightVector, intercept, quantFeatures, catFeatures)

    def runCatImputation(self, quantFeatures, catFeatures, actual):
        le = LabelEncoder()
        le.fit(self.yTrain)
        self.class_encoder = le
        self.yTrain = le.transform(self.yTrain)
        self.yTest = le.transform(self.yTest)
        num_classes = len(le.classes_)

        weights = np.zeros((num_classes, self.xTrain.shape[1]))
        biases = np.zeros(num_classes)
        alpha = self._logRegAlpha

        weights, biases = self.softmaxGradientDescent(self.xTrain, self.yTrain, weights, biases, alpha, num_classes)

        yPred = self.softmaxPredict(self.xTest, weights, biases)
        acc = np.mean(yPred == self.yTest)
        print(f"Test accuracy: {acc:.2f}")

        self.updateDF(actual, weights, biases, quantFeatures, catFeatures, is_classification=True, class_encoder=le)

    def normalizeData(self, x1, x2, cols1, cols2):
        dfQuant = self.normalizeQuant(x1, cols1)
        dfCat = self.normalizeCat(x2, cols2)

        return pd.concat([dfQuant,dfCat], axis=1)


    #Takes quanditative dataframe and standardizes/normalizes data. 
    #Stores means and standard deviations for later use 
    def normalizeQuant(self, x, cols):
        self.quant_means = x.mean()
        self.quant_stds = x.std(ddof=0)
        array = (x - self.quant_means) / self.quant_stds
        return pd.DataFrame(array, columns=cols)


    #Takes Categorical data and returns numerical representation  
    #Stores Encoder and Mode for later use
    def normalizeCat(self, x, cols):
        self.label_encoders = {}   # Dictionary to store LabelEncoder per column
        self.cat_modes = {}        # Store most common value per column (for missing value fill)

        encoded_df = pd.DataFrame(index=x.index)

        for col in cols:
            le = LabelEncoder()
            col_data = x[col].astype(str).str.strip()

            # Fill missing with mode
            mode = col_data.mode()[0]
            self.cat_modes[col] = mode
            col_data = col_data.fillna(mode)

            le.fit(col_data)
            self.label_encoders[col] = le

            encoded_df[col] = le.transform(col_data)

        return encoded_df

    #Computes cost using MSE
    def computeCost(self, x, y, w, b):
        m = x.shape[0]  #number of examples
        cost = 0.0
        for i in range(m):                                
            yHat = np.dot(x[i], w) + b           
            cost = cost + (yHat - y[i])**2       
        cost = cost / (2 * m)                      
        return cost
    

    def computeGradient(self, x, y, w, b):
        m,n = x.shape  #(number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = 0

        for i in range(m):
            err = (np.dot(x[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * x[i, j]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m
            
        return dj_db, dj_dw

    def gradientDescent(self, x, y, w0, b0, alpha):
        maxIters = 1000000
        epsilon = self._linRegEpsilon #minimum difference between iteration errors to trigger satsfaction.
        history = []
        w = copy.deepcopy(w0)
        b = b0
        i = 0

        while(i < maxIters and (len(history)<2 or abs(history[-1]-history[-2]) > epsilon)):
            i += 1

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = self.computeGradient(x, y, w, b) 

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
        
            history.append(self.computeCost(x, y, w, b))

            # Print cost every at intervals 100 times
            if i % 100 == 0:
                print(f"Iteration {i:4d}: Cost {history[-1]:8.2f}   ")
            
        return w, b, history
    

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def softmaxPredict(self, x, weights, biases):
        logits = np.dot(x, weights.T) + biases
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)

    def softmaxGradientDescent(self, x, y, w, b, alpha, num_classes):
        maxIters= 1000
        epsilon = self._logRegEpsilon
        m = x.shape[0]
        history = []
        i = 0

        while(i < maxIters and (len(history)<2 or abs(history[-1]-history[-2]) > epsilon)):
            i += 1
            
            logits = np.dot(x, w.T) + b
            probs = self.softmax(logits)
            y_one_hot = np.eye(num_classes)[y]

            error = probs - y_one_hot 
            grad_w = (1/m) * np.dot(error.T, x)
            grad_b = (1/m) * np.sum(error, axis=0)

            w = w - alpha * grad_w
            b = b - alpha * grad_b

            loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-15), axis=1))
            history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

        print(f"Converged at iteration {i}")

        return w, b

    def predict(self, x, w, b):
        return np.dot(x, w) + b

    # Shows total number of null/NaN values in each feature 
    def showFeatureNAs(self, df):
        print("NaNs in features:\n", df.isnull().sum()) 


    def updateDF(self, actual, weights, bias, quantFeatures, catFeatures, is_classification=False, class_encoder=None):
        for index, row in self.df.iterrows():
            if pd.isnull(row[actual]):
                featureVector = []

                # Quantitative features: fill missing with mean, normalize
                for feature in quantFeatures:
                    val = row[feature]
                    if pd.isnull(val):
                        val = self.df[feature].mean()
                    val = (val - self.quant_means[feature]) / self.quant_stds[feature]
                    featureVector.append(val)

                # Categorical features: fill missing with mode, encode
                for feature in catFeatures:
                    val = row[feature]
                    if pd.isnull(val) or str(val).strip() not in self.label_encoders[feature].classes_:
                        val = self.cat_modes[feature]
                    val = str(val).strip()
                    val = self.label_encoders[feature].transform([val])[0]
                    featureVector.append(val)

                # Prediction
                if is_classification:
                    logits = np.dot(weights, featureVector) + bias
                    probs = self.softmax(np.array([logits]))
                    pred_class = np.argmax(probs)
                    predicted_value = class_encoder.inverse_transform([pred_class])[0]
                else:
                    predicted_value = np.dot(featureVector, weights) + bias

                self.df.loc[index, actual] = predicted_value
