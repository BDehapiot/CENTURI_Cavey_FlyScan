#%% Initialize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import neighbors as nn
from sklearn import naive_bayes as nb
from sklearn.model_selection import train_test_split

#%% Inputs
RootPath = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey/21-02-05_TestMovies/ModelTraining'
DataFilename = '/ClassData_Pi00-01-02.csv'
TestDataFilename = '/ClassData_OldTest.csv'

#%% Open data
class_data = pd.read_csv(RootPath+DataFilename, sep=';') 
test_data = pd.read_csv(RootPath+TestDataFilename, sep=';') 
X = class_data[['area', 'grd_SD', 'grd_Quant']] # data without class
y = class_data[['class']] # class only
X_test = test_data

#%% Train model

# # Generate random training dataset
# Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.5, random_state=1) 

clf = nb.GaussianNB()  
clf.fit(X, y)
test_class = clf.predict(X_test)




# # Determine KNeighborsClassifier hyperparameters
# krange = np.arange(1, 20, dtype=int)
# erreurs = np.zeros(len(krange), dtype=float)
# i=0
# for k in krange:
#     clf = nn.KNeighborsClassifier(k) 
#     clf.fit(Xtrain,ytrain)
#     erreurs[i]=1-clf.score(Xtest,ytest)   
#     i=i+1
# plt.plot(krange,erreurs)

# # Train KNeighborsClassifier
# clf = nn.KNeighborsClassifier(6) 
# clf.fit(Xtrain,ytrain)







