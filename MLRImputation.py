# The purpose of this class will be to use multiple linear regression 
# to predict the fuel type of the fire (type of trees/vegetation) based on the following factors
# Quantitative Data:
# - fire_spread_rate
# - temperature
# - relative_humidity
# - wind_direction
# - wind_speed
# - 
# Categorical Data:
# - fire_type
# - fire_position_on_slope
# - weather_conditions_over_fire
# - size_class

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
import copy

#Multiple Linear Regression Imputation
class MLRImputation:

    #args - original pandas dataframe, quantitative features, categorical features, actual feature to imputate vales
    #returns - data frame with null values replaced with imputated values. 
    def __init__(self, df, quantFeatures, catFeatures):

        self.df = df #includes only features being used

        #runs imputation on each quantitative feature and updates dataframe NA cells with imputated values  
        for actual in quantFeatures:
            rQuantFeatures = [feature for feature in quantFeatures if feature != actual]
            
            # # skips imputation if no NA values are detected
            # if(self.df[actual].isnull().sum() > 0):
                #Runs imputation 
            print(f"Running Imputation on :{actual}")
            self.runImputation(df, rQuantFeatures, catFeatures, actual)
        


    def runImputation(self, df, quantFeatures, catFeatures, actual):
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
        xTrain = self.normalizedDF[div].values
        xTest = self.normalizedDF[~div].values
        yTrain = self.dfActual[div].values.flatten()
        yTest = self.dfActual[~div].values.flatten()

    
        # print(f"training data:\n {xTrain}")
        # print(f"testing data:\n {xTest}")


        # Set initial values for gradient descent
        w0 = np.zeros(xTrain.shape[1])
        b0 = 0
        alpha = 0.01 #Learning rate

        weightVector, intercept, history = self.gradientDescent(xTrain, yTrain, w0, b0, alpha)
        print(f"intercept, weights found by gradient descent: {intercept:0.2f}, {weightVector} ")
        # print(f"history: {history}")

        yPred = self.predict(xTest, weightVector, intercept)
        mse = np.mean((yPred - yTest) ** 2)

        print(f"Test MSE: {mse:.2f}")

        self.updateDF(actual, intercept, weightVector, quantFeatures, catFeatures)



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
        m,n = x.shape           #(number of examples, number of features)
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
        epsilon = 0.001 #minimum difference between iteration errors to trigger satsfaction.
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
    

    def predict(self, x, w, b):
        return np.dot(x, w) + b

    # Shows total number of null/NaN values in each feature 
    def showFeatureNAs(self, df):
        print("NaNs in features:\n", df.isnull().sum()) 


    #Note: if, in addition to the actual being null, another feature is null, we use the mean of that feature data 
    def updateDF(self, actual, intercept, weightVector, quantFeatures, catFeatures):
        
        for index, row in self.df.iterrows():
            if pd.isnull(row[actual]):
                featureVector = []

                # Ensures proper order of vector variables
                # Assigns temporary variable mean to vector is case of null value (mode for categorical)
                # Normalizes each value using same std, mean, mode, and encoder used in normalization for test data
                for feature in quantFeatures:
                    val = row[feature]
                    if pd.isnull(val):
                        val = self.df[feature].mean()
                        val = (val - self.quant_means[feature]) / self.quant_stds[feature]
                    featureVector.append(val)

                for feature in catFeatures:
                    val = row[feature]
                    if pd.isnull(val) or str(val).strip() not in self.label_encoders[feature].classes_:
                        val = self.cat_modes[feature]

                    val = str(val).strip()
                    val = self.label_encoders[feature].transform([val])[0]
                    featureVector.append(val)


                self.df.loc[index, actual] = (np.dot(featureVector, weightVector) + intercept)


