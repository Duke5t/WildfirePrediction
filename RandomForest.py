from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Bagging - Bootstrap aggregation : manufacturing different data 
# Bagging - Manufactures data with same size as original data but with potential duplicates/overrepresentation (Bags)
# Bagging works with random sampling with replacement
# Bagging - Compliment of Bag 1 and original data is called "Out Of Bag Data" (data that is in Original and not in Bag1)

# Feature Randomization - Each bag is assigned a subset of features (formula: num features in subset = sqrt(total features))
# Reference : https://www.youtube.com/watch?v=IYXeCgMQ4to&ab_channel=DavidLanger


# 

class RandomForest:

    def __init__(self, dataFrame, quantFeat, catFeat, predFeat):
        pass

    #Create labels
    #Create RandomForestClassifier

    # How do I choose how many estimators for random forest (n_estimators = how many trees)?
    # OOB score uses OOB data as test data

    #Create the random forest classifier
    fireRF = RandomForestClassifier(n_estimators = 100, oob_score = True)

    

    #Train the classifier
    fireRF.fit(fireX,fireY) 
    #fireX is quant variables + OneHotEncoded cat variables
    #fireY is predicted variable encoded (fireY = labelEncoder.fit_transform( [##single predictFeat] ))


    print(f'Accuracy with OOB daa: {fireRF.oob_score_:.4f}') #Prints accuracy when compared to OOB Data (Out Of Bag)



    oobPreds = pd.DataFrame(fireRF.oob_decision_function_, columns = labelEncoder.classes_) #References label encoder for y data
    oobPreds['Label'] = 0
    #oobPreds.loc[oobPreds[insert true prediction value here] > 0.5, 'Label'] = 1

    #plots confusion matrix (shows TP, FP, TN, FN) (helpful for visualization) (helpful for F1?)
    cm = confusion_matrix(fireY, oobPreds['Label'])
    cmd = ConfusionMatrixDisplay(cm, display_labels = labelEncoder.classes_) #References label encoder for y data
    cmd.plot()

