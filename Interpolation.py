# The purpose of this class will be to use multiple linear regression 
# to predict the fuel type of the fire (type of trees/vegetation) based on the following factors



import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder
import copy

#Multiple Linear Regression Interpolation
class Interpolation:

    _linRegAlpha = 0.01 ##Learning Rate for Linear Regression
    _linRegEpsilon = 0.001 ##Convergence Rate for Linear Regression
    _logRegAlpha = 0.01 ##Learning Rate for Logisitc Regression
    _logRegEpsilon = 0.001 ##Convergence Rate for Logistic Regression
    _testTrainSplit = 0.85 #.8 = 80% Train / 20% Test

    #args - original pandas dataframe, quantitative features, categorical features, actual feature to impute values
    #returns - data frame with null values replaced with interpolated values. 
    def __init__(self, df, quantFeatures, catFeatures):
        print("\n---INTERPOLATION---\n")

        self.df = df.copy() #includes only features being used

        #runs Interpolation on each quantitative feature and updates dataframe NA cells with interpolated values  
        for actual in quantFeatures:
            rQuantFeatures = [feature for feature in quantFeatures if feature != actual]
            print(f"Running Interpolation on :{actual}")
            self.prepData(df, rQuantFeatures, catFeatures, actual)
            self.runQuantInterpolation(rQuantFeatures, catFeatures, actual)

        #runs Interpolation on each categorical feature and updates dataframe NA cells with interpolated values  
        for actual in catFeatures:
            rCatFeatures = [feature for feature in catFeatures if feature != actual]
            print(f"Running Interpolation on :{actual}")
            self.prepData(df, quantFeatures, rCatFeatures, actual, True)
            self.runCatInterpolation(quantFeatures, rCatFeatures, actual)

        # Confirms final df has no null values remaining
        # print(self.df.shape[0])
        self.showFeatureNAs(self.df)
        # print(self.df.head(10))

    # Drops rows with any NA values
    # Normalizes data
    # Sets test and training subsets (80/20)
    def prepData(self, df, quantFeatures, catFeatures, actual, isActualCategorical = False):
        # Drops rows which contain NA values in any selected column
        self.dfNoNull = df.copy().dropna().reset_index(drop=True) 
        print(f"# of Rows in Feature with null vals: {self.df[actual].isnull().sum()}")
        # Creates copies of df for quantitative, catergorical features and the intended/actual feature to impute
        self.dfQuant = self.dfNoNull[quantFeatures].copy()
        self.dfCat = self.dfNoNull[catFeatures].copy()
        self.dfActual = self.dfNoNull[actual].copy()

        self.normalizedDF = self.normalizeData(self.dfQuant, self.dfCat, quantFeatures, catFeatures)

        # Divide 80/20 train/test data
        self.div = np.random.rand(len(self.dfNoNull)) < self._testTrainSplit
        self.xTrain = self.normalizedDF[self.div].values
        self.xTest = self.normalizedDF[~self.div].values

        ##Flattens quant Y variable, strips if the 'actual' variable is categorical
        if isActualCategorical:
            self.yTrain = self.dfActual[self.div].astype(str).str.strip().values
            self.yTest = self.dfActual[~self.div].astype(str).str.strip().values
        else:
            self.yTrain = self.dfActual[self.div].values.flatten()
            self.yTest = self.dfActual[~self.div].values.flatten()


    def runQuantInterpolation(self, quantFeatures, catFeatures, actual):
        # Set initial values for gradient descent
        w0 = np.zeros(self.xTrain.shape[1])
        b0 = 0
        alpha = self._linRegAlpha #Learning rate

        weightVector, intercept, history = self.gradientDescent(self.xTrain, self.yTrain, w0, b0, alpha)
        # print(f"intercept, weights found by gradient descent: {intercept:0.2f}, {weightVector} ")
        # print(f"history: {history}")

        yPred = self.predict(self.xTest, weightVector, intercept)
        mse = np.mean((yPred - self.yTest) ** 2)
        print(f"Test MSE: {mse:.2f}")

        self.updateDF(actual, weightVector, intercept, quantFeatures, catFeatures)


    # Run Categorical Interpolation
    # Uses logistic regression with softmax function
    # Encodes Labels using OneHotEncoder
    def runCatInterpolation(self, quantFeatures, catFeatures, actual):
        # Encode labels using LabelEncoder
        le = LabelEncoder()
        allLabels = np.concatenate([self.yTrain, self.yTest]) ##Ensures we don't miss any labels that are only in test data
        le.fit(allLabels)
        self.classEncoder = le
        self.yTrain = le.transform(self.yTrain)
        self.yTest = le.transform(self.yTest)
        numClasses = len(le.classes_)

        weights = np.zeros((numClasses, self.xTrain.shape[1]))
        biases = np.zeros(numClasses)
        alpha = self._logRegAlpha

        # Train softmax regression via gradient descent
        weights, biases = self.softmaxGradientDescent(self.xTrain, self.yTrain, weights, biases, alpha, numClasses)

        # Predict and calculate test accuracy
        yPred = self.softmaxPredict(self.xTest, weights, biases)
        acc = np.mean(yPred == self.yTest)
        print(f"Number of classes: {numClasses} \nTest accuracy: {acc:.2f}")

        # Update dataframe with predicted values
        self.updateDF(actual, weights, biases, quantFeatures, catFeatures, isClassification=True, classEncoder=le)

    #Takes quanditative and categorical data and standardizes/normalizes/encodes data
    def normalizeData(self, x1, x2, cols1, cols2):
        dfQuant = self.normalizeQuant(x1, cols1)
        dfCat = self.normalizeCat(x2, cols2)
        return pd.concat([dfQuant,dfCat], axis=1)

    #Takes quantitative dataframe and standardizes/normalizes data. 
    #Stores means and standard deviations for later use 
    def normalizeQuant(self, x, cols):
        self.quantMeans = x.mean()
        self.quantStds = x.std(ddof=0)
        array = (x - self.quantMeans) / self.quantStds
        return pd.DataFrame(array, columns=cols)

    #Takes Categorical data and returns One-Hot Encoded representation  
    #Stores Encoder and Mode for later use
    def normalizeCat(self, x, cols):
        self.catModes = {}        # Store most common value per column (for missing value fill)
        self.oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Set up OHE

        # Find mode values for each feature
        for col in cols:      
            mode = x[col].astype(str).str.strip().mode(dropna=True)[0]
            self.catModes[col] = mode

        #Transforms features to encodedArray using OHE
        encodedArray = self.oneHotEncoder.fit_transform(x[cols])
        featureNames = self.oneHotEncoder.get_feature_names_out(cols)

        return pd.DataFrame(encodedArray, columns=featureNames, index=x.index)


    #Uses Mean Squared Error as cost function
    def computeCost(self, x, y, w, b):
        m = x.shape[0]

        #Uses vectorization to compute cost (MSE)
        yHat = np.dot(x, w) + b
        cost = np.mean((yHat - y) ** 2) / 2

        return cost


    def computeGradient(self, x, y, w, b):
        m = x.shape[0]  # number of examples

        # Compute prediction error
        error = np.dot(x, w) + b - y  # shape: (m,)

        # Compute gradient with respect to weights (dj_dw)
        dj_dw = np.dot(x.T, error) / m  # shape: (n,)

        # Compute gradient with respect to bias (dj_db)
        dj_db = np.mean(error)

        return dj_db, dj_dw


    def gradientDescent(self, x, y, w0, b0, alpha):
        maxIters = 250000
        epsilon = self._linRegEpsilon #minimum difference between iteration errors to trigger satsfaction.
        history = []
        w = copy.deepcopy(w0)
        b = b0
        i = 0

        while(i < maxIters and (len(history)<2 or abs(history[-1]-history[-2]) > epsilon)):
            i += 1
            dj_db, dj_dw = self.computeGradient(x, y, w, b) # Calculate the gradient and update the parameters
            
            #Update weights and biases
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            history.append(self.computeCost(x, y, w, b))

        return w, b, history

    #Uses logistic regression with softmax function
    def softmax(self, z):
        # Compute the softmax function in a numerically stable way.
        # Subtract max for each row to avoid large exponents that cause overflow.
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 

        #Normalize
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def softmaxPredict(self, x, weights, biases):
        # Compute raw model outputs (logits)
        logits = np.dot(x, weights.T) + biases

        #Convert logits to probabilities using softmax 
        probs = self.softmax(logits)

        #Return max probability (predicted class)
        return np.argmax(probs, axis=1)

    def softmaxGradientDescent(self, x, y, w, b, alpha, numClasses):
        maxIters= 25000
        epsilon = self._logRegEpsilon
        m = x.shape[0]
        history = []
        i = 0

        while(i < maxIters and (len(history)<2 or abs(history[-1]-history[-2]) > epsilon)):
            i += 1
            #Compute logits (raw output)
            logits = np.dot(x, w.T) + b

            #Compute predicted probabilities
            probs = self.softmax(logits)

            #One Hot Encoded True Labels
            yOneHot = np.eye(numClasses)[y]
            error = probs - yOneHot 

            #Compute gradients
            grad_w = (1/m) * np.dot(error.T, x)
            grad_b = (1/m) * np.sum(error, axis=0)

            #Update weights and biases
            w = w - alpha * grad_w
            b = b - alpha * grad_b

            # Cross-entropy loss (log loss) (1e-15 prevents log(0) error)
            loss = -np.mean(np.sum(yOneHot * np.log(probs + 1e-15), axis=1))
            history.append(loss)

        print(f"Converged at iteration {i}")
        return w, b

    def predict(self, x, w, b):
        return np.dot(x, w) + b

    # Shows total number of null/NaN values in each feature 
    def showFeatureNAs(self, df):
        print("NaNs in features:\n", df.isnull().sum()) 

    # Updates dataframe with predicted values (numerical or categorical)
    def updateDF(self, actual, weights, bias, quantFeatures, catFeatures, isClassification=False, classEncoder=None):
        for index, row in self.df.iterrows():
            if pd.isnull(row[actual]):
                featureVector = []

                # Quantitative features: fill missing with mean, normalize
                for feature in quantFeatures:
                    val = row[feature]
                    if pd.isnull(val):
                        val = self.df[feature].mean()
                    val = (val - self.quantMeans[feature]) / self.quantStds[feature]
                    featureVector.append(val)

                # Categorical features: fill missing with mode, encode using OneHotEncoder
                catVals = {}
                for feature in catFeatures:
                    val = row[feature]
                    if pd.isnull(val) or str(val).strip() not in self.oneHotEncoder.categories_[catFeatures.index(feature)]:
                        val = self.catModes[feature]
                    catVals[feature] = str(val).strip()

                catDf = pd.DataFrame([catVals])
                catEncoded = self.oneHotEncoder.transform(catDf[catFeatures])
                featureVector.extend(catEncoded.flatten())

                # Prediction
                if isClassification:
                    logits = np.dot(weights, featureVector) + bias
                    probs = self.softmax(np.array([logits]))
                    predClass = np.argmax(probs)
                    predictedValue = classEncoder.inverse_transform([predClass])[0]
                else:
                    predictedValue = np.dot(featureVector, weights) + bias

                # Set predicted val in DF cell
                self.df.loc[index, actual] = predictedValue
