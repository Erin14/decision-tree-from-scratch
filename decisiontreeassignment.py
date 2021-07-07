import sys
import numpy as np
from math import log2

__author__ = "Ruiling Chen"
__version__ = "0.1.0"

def MaxIG(dataset):
    '''
    Calculates the Information Gain (IG) for each features in the dataset and choose
    the one with the largest IG, return it.

    Parameters:
        dataset (np.array): The dataset that needs to be calculate.

    Returns:
        newFeat (str): The feature name with the maximum IG. ("yes" or "no" if reach 
            the leave node.)

        newFeatDic (dictionary): The dictionary contains all the possible values of 
            newFeat and their counts in the dataset. ("None" if reach the leave node.)

        newFeatIndex (int): The index of newFeat in the features line of dataset.
    '''

    # split the dataset to be features and data
    features = dataset[0][0:-1]
    dataset = dataset[1:]
    data_size = len(dataset)

    # calculate the probablities of conclusion
    conclusion1 = 0
    for dataIndex in range(data_size):
        if dataset[dataIndex][-1] == "yes":
            conclusion1 += 1
    Prob_conclu1 = conclusion1 / data_size
    Prob_conclu0 = 1 - Prob_conclu1

    # return the label if the conclusion of the dataset are the same, or return the
    # label who happens more if the dataset only contains the conclusion column.
    if Prob_conclu0 == 1 or (len(dataset[0]) == 1 and Prob_conclu0 >= 0.5):
        return "no", None, -1
    elif Prob_conclu1 == 1 or (len(dataset[0]) == 1):
        return "yes", None, -1

    # the entropy of conclusion: H(conclusion)
    H_conclu = -(Prob_conclu1 * log2(Prob_conclu1) + Prob_conclu0 * log2(Prob_conclu0))

    newIG = 0
    newFeat = ""
    newFeatDic = {}
    newFeatIndex = 0

    for featureIndex in range(len(features)):
        
        # create the dictionary of the feature
        featDic = {}
        for dataIndex in range(data_size):
            value = dataset[dataIndex][featureIndex]
            if value not in featDic.keys():
                featDic[value] = 0
            featDic[value] += 1

        # calculate the entropy of conclusion given feature: H(conclusion|feature)
        H_ConcluGivenFeat = 0

        for value in featDic.keys():
            count_ValAndCon1 = 0
            for dataIndex in range(data_size):
                if (dataset[dataIndex][featureIndex] == value) and (dataset[dataIndex][-1] == "yes"):
                    count_ValAndCon1 += 1

            Prob_Con1GivenVal = count_ValAndCon1/featDic[value]
            
            Prob_1GV = Prob_Con1GivenVal # shorter name

            if (Prob_1GV < 10e-6) or (1-Prob_1GV < 10e-6):
                H_ConcluGivenVal = 0
            else:
                H_ConcluGivenVal = -(Prob_1GV * log2(Prob_1GV) + (1 - Prob_1GV) *
                                     log2(1 - Prob_1GV))

            Prob_val = featDic[value] / data_size
            H_ConcluGivenFeat += Prob_val * H_ConcluGivenVal

        IG = H_conclu - H_ConcluGivenFeat

        if IG > newIG:
            newIG = IG
            newFeat = features[featureIndex]
            newFeatDic = featDic
            newFeatIndex = featureIndex
            
    if newIG == 0:
        if Prob_conclu0 >= 0.5:
            return "no", None, -1
        else:
            return "yes", None, -1
    
    return newFeat, newFeatDic, newFeatIndex


def createTree(dataset, max_depth):
    '''
    Create a decision tree depends on the given dataset.

    Parameters:
        dataset (np.array): The dataset that needs to create decision tree.
        max_depth (int): The maximum tree depth (-1 if no limited depth).

    Returns:
        node (str): Return the label "yes" or "no" when reach the leave node, or
        myTree (dic): Return the tree as a dictionary.
    '''
    # when reach the setting max depth
    if max_depth == 0:
        sub_dataset = np.delete(dataset, range(len(dataset[0])-1), axis = 1)
        node, nodeDic, index = MaxIG(sub_dataset)
        return node

    data_size = len(dataset) - 1
    node, nodeDic, nodeIndex = MaxIG(dataset)
    if nodeIndex == -1:
        return node

    myTree = {node:{}}
    # create the sub dataset depends on the conditions of node feature
    for condition in nodeDic.keys():
        sub_dataset = dataset[0]
        for dataIndex in range(data_size):
            if dataset[dataIndex + 1][nodeIndex] == condition:
                sub_dataset = np.vstack((sub_dataset, dataset[dataIndex + 1]))
        sub_dataset = np.delete(sub_dataset, nodeIndex, axis = 1)
            
        myTree[node][condition] = createTree(sub_dataset, max_depth - 1)

    return myTree

def testResult(myTree, features, data):
    '''
    Based on the decision tree, get the conclusion of the given data.

    Parameters:
        myTree (dic): The decision tree.
        features (list): The list of features name of the data.
        data (list): The list of one test data.
        
    Returns:
        myTree (str): Return the label "yes" or "no" when reach the leave node.
    '''

    if type(myTree) == str:
        return myTree

    for feature in features:
        if feature == list(myTree.keys())[0]:
            featIndex = features.index(feature)
            features.pop(featIndex) # delete the feature from the feature list
            value = data.pop(featIndex) # delete the value from the data
            subTree = myTree[feature][value]
            return testResult(subTree, features, data)
        
    
def test(myTree, dataset):
    '''
    Compare the test result and the actual result for the dataset, calculate the 
    accuracy.
    
    Parameters:
        myTree (dic): The decision tree created by training data.
        dataset (np.array): The dataset we want to calculate the accuracy.

    Returns:
        (float): The accuracy of the dataset given decision tree.
    ''' 
    datasize = len(dataset) - 1
    myResults = [] # test result
    features = list(dataset[0])
    for i in range(datasize):
        data = list(dataset[i+1])
        myResults += [testResult(myTree, features[:], data),]

    actualResults = list(dataset[1:, -1])
    sameResult = 0
    for i in range(datasize):
        if actualResults[i] == myResults[i]:
            sameResult += 1

    return sameResult/datasize

    
if __name__ == '__main__':

    # deal with the input parameters
    train_datafile = sys.argv[1]
    max_depth = -1
    test_datafile = None

    if len(sys.argv) == 3:
        try:
            max_depth = int(sys.argv[2])
        except:
            test_datafile = sys.argv[2]

    elif len(sys.argv) == 4:
        max_depth = int(sys.argv[2])
        test_datafile = sys.argv[3]

    train_data = np.genfromtxt(train_datafile, dtype='str')
    myTree = createTree(train_data, max_depth)
    print("The decision tree is:", myTree)
    
    train_accuracy = test(myTree, train_data)
    print("The accuracy for the training set is {}.".format(train_accuracy))
    
    if test_datafile is not None:
        test_data = np.genfromtxt(test_datafile, dtype='str')
        test_accuracy = test(myTree, test_data)
        print("The accuracy for the testing set is {}.".format(test_accuracy))

