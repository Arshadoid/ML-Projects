# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:14:15 2022

@author: Arshad Mehtiyev
"""

#Importing main libraries

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn import tree, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier

# Training and testing of Decision Tree Classifier

def DT_classifier(X_train_flt,
                  Y_train, 
                  X_validation_flt, 
                  Y_validation,
                  X_test_flt,
                  Y_test):
    # Performance check on Validation set
    # Performance with "gini" criterion with best split
    
    clf_DT=tree.DecisionTreeClassifier(criterion='gini')
    clf_DT = clf_DT.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_DT.predict(X_validation_flt)
    metr= metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Decision Tree with 'gini' and best split criterion")
    print("Accuracy: ", metr)
    
    
    #Performance with "entropy" criterion wiht best split
    
    clf_DT=tree.DecisionTreeClassifier(criterion='entropy')
    clf_DT = clf_DT.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_DT.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Decision Tree with 'entropy' and best split criterion")
    print("Accuracy: ", metr)
    
    #Performance with "entropy" criterion with random split
    
    clf_DT=tree.DecisionTreeClassifier(criterion='entropy',splitter='random')
    clf_DT = clf_DT.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_DT.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Decision Tree with 'entropy' and random split criterion")
    print("Accuracy: ", metr)

    # Performance with "entropy" criterion with random split with depth of 20
    
    clf_DT=tree.DecisionTreeClassifier(criterion='entropy',splitter='random', 
                                       max_depth=20)
    clf_DT = clf_DT.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_DT.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Decision Tree with 'entropy' and random split criterion")
    print("and depth of 20")
    print("Accuracy: ", metr)
    
    # Performance check on Test set with the best performance hyperparameters

    clf_DT=tree.DecisionTreeClassifier(criterion="entropy", splitter='random')
    clf_DT = clf_DT.fit(X_train_flt,Y_train)
    Y_test_pred = clf_DT.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred)    
    print("Performance of Decision Tree on Test set with the best performance") 
    print("hyperparameters")
    print("Accuracy: ", metr)


# Training and testing of K-Nearest Neighbors Classifier 
def KNN_classifier(X_train_flt,
                  Y_train, 
                  X_validation_flt, 
                  Y_validation,
                  X_test_flt,
                  Y_test):
    
    # Performance check on Validation set

    # With 1-Nearest Neighbor
    
    clf_KNN=KNeighborsClassifier(n_neighbors=1)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=1") 
    print("Accuracy: ", metr)
    
    # With 3-Nearest Neighbors
    
    clf_KNN=KNeighborsClassifier(n_neighbors=3)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=3") 
    print("Accuracy: ", metr)
    
    # With 5-Nearest Neighbors
    
    clf_KNN=KNeighborsClassifier(n_neighbors=5)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=5") 
    print("Accuracy: ", metr)
    
    # With 15-Nearest Neighbors  
    
    clf_KNN=KNeighborsClassifier(n_neighbors=15)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=15") 
    print("Accuracy: ", metr)
    
    # With 5-Nearest Neighbors, 'distance' type weight function and p=1 power 
    # parameter for Minkowski metric(Manhattan distance)
    
    clf_KNN=KNeighborsClassifier(n_neighbors=5, weights='distance',p=1)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=5, weidght='distance' ")
    print("and p=1 parameters")
    print("Accuracy: ", metr)
    
    # With 5-Nearest Neighbors and  p=1 power parameter for Minkowski metric
    # (Manhattan distance)
    
    clf_KNN=KNeighborsClassifier(n_neighbors=5,p=1)
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=5 and p=1 parameters")
    print("Accuracy: ", metr)

    # With 5-Nearest Neighbors and 'distance' type weight function
    
    clf_KNN=KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=5 and weights='distance'")
    print("parameters")
    print("Accuracy: ", metr)

    # With 3-Nearest Neighbors and 'distance' type weight function
    
    clf_KNN=KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_KNN.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of KNN on Validation set with K=3 and weights='distance'")
    print("parameters")
    print("Accuracy: ", metr)

    # Performance check on Test set with the best performance parameters
    
    clf_KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf_KNN = clf_KNN.fit(X_train_flt,Y_train)
    Y_test_pred = clf_KNN.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred)
    print("Performance of KNN on test set with the best performance parameters")
    print("Accuracy: ", metr)



# Training Gaussian Niave Bayes Classifier


def Gaussian_NB(X_train_flt,
                  Y_train, 
                  X_validation_flt, 
                  Y_validation,
                  X_test_flt,
                  Y_test):
    

    # Training and performance check on Validation set

    # With var_smooth = 1e-9 (default) parameter
    
    clf_GNB=GaussianNB()
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with the default parameters")
    print("Accuracy: ", metr)

    # With var_smooth = 1e-7
    
    clf_GNB=GaussianNB(var_smoothing=1e-7)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with var_smoothing=1e-7 parameter")
    print("Accuracy: ", metr)

    # With var_smooth = 1e-5
    
    clf_GNB=GaussianNB(var_smoothing=1e-5)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with var_smoothing=1e-3 parameter")
    print("Accuracy: ", metr)

    # With var_smooth = 1e-3
    
    clf_GNB=GaussianNB(var_smoothing=1e-3)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with var_smoothing=1e-3 parameter")
    print("Accuracy: ", metr)
    
    # With var_smooth = 1e-2
    
    clf_GNB=GaussianNB(var_smoothing=1e-2)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with var_smoothing=1e-2 parameter")
    print("Accuracy: ", metr)

    # With var_smooth = 1e-1
    clf_GNB=GaussianNB(var_smoothing=1e-1)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_GNB.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of Gaussian NB on Validation set with var_smoothing=1e-1 parameter")
    print("Accuracy: ", metr)
    
    # Performance check on Test set with the best performance hyperparameters 
    
        
    clf_GNB=GaussianNB(var_smoothing=1e-1)
    clf_GNB = clf_GNB.fit(X_train_flt,Y_train)
    Y_test_pred = clf_GNB.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred)
    print("Performance of Gaussian NB on Test set with var_smoothing=1e-1 parameter")
    print("Accuracy: ", metr)
    
    
# Training Support Vector Machine (SVC and LinearSVC) Classifier   
def SVM_classifier(X_train_flt,
                  Y_train, 
                  X_validation_flt, 
                  Y_validation,
                  X_test_flt,
                  Y_test):
    
    # Performance check on Validation set
    
    # SVC With default parameters
    
    clf_SVM=SVC()
    clf_SVM = clf_SVM.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_SVM.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of SVC on Validation set with default parameters")
    print("Accuracy: ", metr)

    # SVC With polynomial kernel
    
    clf_SVM=SVC(kernel='poly')
    clf_SVM = clf_SVM.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_SVM.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of SVC on Validation set with 'poly' parameter")
    print("Accuracy: ", metr)

    # LinearSVC with default parameters
    
    clf_SVM2=LinearSVC()
    clf_SVM2 = clf_SVM2.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_SVM.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of linearSVC on Validation set with default parameters")
    print("Accuracy: ", metr)
    
    # LinearSVC with  dual=False  parameter
    
    clf_SVM2=LinearSVC(dual=False)
    clf_SVM2 = clf_SVM2.fit(X_train_flt,Y_train)
    Y_valid_pred = clf_SVM.predict(X_validation_flt)
    metr = metrics.accuracy_score(Y_validation,Y_valid_pred)
    print("Performance of linearSVC on Validation set with daul=False parameter")
    print("Accuracy: ", metr)

    # Performance check of SVC on Test set with the default hyperparameter
    clf_SVM=SVC()
    clf_SVM = clf_SVM.fit(X_train_flt,Y_train)
    Y_test_pred = clf_SVM.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred)
    print("Performance of SVM(SVC) on Test set with default parameters")
    print("Accuracy: ", metr)

# Ensemble Classifier with hard voting




def Ensemble_classifier_hv(X_train_flt,
                  Y_train, 
                  X_validation_flt, 
                  Y_validation,
                  full_X_train,
                  full_Y_train,
                  X_test_flt,
                  Y_test,
                  X_test):

    #  Defining classifiers with their best hyperparameters to be used in Ensemble 
    clf_DT=tree.DecisionTreeClassifier(criterion="entropy", splitter='random')
    clf_KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf_GNB=GaussianNB(var_smoothing=1e-1)
    clf_SVM=SVC()
    
    # Building Ensemble Classifier with "Hard" voting
    
    hard_voting_clf = VotingClassifier(
        estimators=[
            ('dt', clf_DT),
            ('knn', clf_KNN),
            ('gnb', clf_GNB),
            ('svm', clf_SVM)],
        voting='hard'    
    )
    
    # Training and testing hard voting classifier on Test Set
    
    hard_voting_clf.fit(X_train_flt, Y_train)
    Y_test_pred = hard_voting_clf.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred)
    print("Performance of Ensemble Classifier with hard voting")
    print("Accuracy: ", metr)
    
    
    # Building Ensemble Classifier with "Hard" voting for full training set
        
    hard_voting_clf_2 = VotingClassifier(
        estimators=[
            ('dt', clf_DT),
            ('knn', clf_KNN),
            ('gnb', clf_GNB),
            ('svm', clf_SVM)],
        voting='hard'    
    )
    
    
    # Training and testing hard voting classifier on Test Set
    
    hard_voting_clf_2.fit(full_X_train, full_Y_train)
    Y_test_pred_full = hard_voting_clf_2.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred_full)
    print("Performance of Ensemble Classifier with hard voting, trained on full training set")
    print("Accuracy: ", metr)

    # extra code to show wrongly predicted targets for ensemble classifier 
    # for hard voting  
    # this code can be called inside of the classifier at the end to show the 
    # final predication results
    # but it needs to be adjusted 

    
    #building list of ra
    wrong_Pred_List=[]

    for i in range(10000):
        if Y_test_pred_full[i]!=Y_test[i]:
            wrong_Pred_List.append(i)
    
    
    n_rows=4
    n_cols=5
    pred_numbers=[]
    img_id = []
    for i in range(1,n_rows*n_cols+1):
        idx = random.randint(0,len(wrong_Pred_List)-1)
        im_idx =wrong_Pred_List[idx]
        img_id.append(im_idx)
        pixels=X_test[im_idx]
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(pixels, cmap='gray')
        plt.axis('off')
        pred_numbers.append(Y_test_pred_full[im_idx])
        
    plt.show()
    arr=np.array(pred_numbers)
    arr.shape = (4,5)
    print(arr)
    img_id


    
# Ensemble Classifier with soft voting

def Ensemble_classifier_sv(X_train_flt,
                           Y_train, 
                           X_validation_flt, 
                           Y_validation,
                           X_test_flt,
                           Y_test):
    
    #  Defining classifiers with their best hyperparameters to be used in Ensemble 
    clf_DT=tree.DecisionTreeClassifier(criterion="entropy", splitter='random')
    clf_KNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf_GNB=GaussianNB(var_smoothing=1e-1)
    clf_SVM=SVC()
    
    # Building Ensemble Classifier with "Soft" voting
    soft_voting_clf = VotingClassifier(
        estimators=[
            ('dt', clf_DT),
            ('knn', clf_KNN),
            ('gnb', clf_GNB),
            ('svm', clf_SVM)],
        voting='soft'    
    )
    
    soft_voting_clf.named_estimators['svm'].probability=True

    
    # Training and testing soft voting classifier on Test Set
    soft_voting_clf.fit(X_train_flt, Y_train)
    Y_test_pred_soft = soft_voting_clf.predict(X_test_flt)
    metr = metrics.accuracy_score(Y_test,Y_test_pred_soft)
    print("Performance of Ensemble Classifier with soft voting")
    print("Accuracy: ", metr)
    

def main():
    # Loading MNIST data 

    data = mnist.load_data()
    (trainX, trainy), (X_test, Y_test) = data

    # Data split into Train, Valdiation and Test sets, and flattining of the
    # feature vectors

    X_train = trainX[:55000,:,:]
    X_validation = trainX[55000:,:,:]
    Y_train = trainy[:55000]
    Y_validation = trainy[55000:]

    # Flattining of the feature vectors

    X_train_flt = X_train.reshape(X_train.shape[0],X_train[0,:,:].size)
    X_validation_flt = X_validation.reshape(X_validation.shape[0], 
                                            X_validation[0,:,:].size)
    X_test_flt = X_test.reshape(X_test.shape[0], X_test[0,:,:].size)

    #merging train and validation sets to train classifier at the end

    full_X_train=np.concatenate((X_train_flt,X_validation_flt))
    full_Y_train=np.concatenate((Y_train,Y_validation))
    
    
    
    # Each function below calls for the previously defined classifier.
    # To run it each seperately, just uncomment it
    
    # DT_classifier(X_train_flt,
    #                   Y_train, 
    #                   X_validation_flt, 
    #                   Y_validation,
    #                   X_test_flt,
    #                   Y_test)
    
    # KNN_classifier(X_train_flt,
    #                   Y_train, 
    #                   X_validation_flt, 
    #                   Y_validation,
    #                   X_test_flt,
    #                   Y_test)
    
    # Gaussian_NB(X_train_flt,
    #                   Y_train, 
    #                   X_validation_flt, 
    #                   Y_validation,
    #                   X_test_flt,
    #                   Y_test)
    
    # SVM_classifier(X_train_flt,
    #                   Y_train, 
    #                   X_validation_flt, 
    #                   Y_validation,
    #                   X_test_flt,
    #                   Y_test)
    
    # Ensemble_classifier_hv(X_train_flt,
    #                   Y_train, 
    #                   X_validation_flt, 
    #                   Y_validation,
    #                   full_X_train,
    #                   full_Y_train,
    #                   X_test_flt,
    #                   Y_test,
    #                   X_test)
    
    # Ensemble_classifier_sv(X_train_flt,
    #                            Y_train, 
    #                            X_validation_flt, 
    #                            Y_validation,
    #                            X_test_flt,
    #                            Y_test)


main()