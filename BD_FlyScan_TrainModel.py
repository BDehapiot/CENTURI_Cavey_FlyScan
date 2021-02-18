#%% Initialize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import neighbors as nn
from sklearn.model_selection import train_test_split

#%% Inputs
RootPath = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey/21-02-05_TestMovies/ModelTraining'
DataFilename = '/ClassData_Pi00-01-02.csv'

#%% Open data
data_class = pd.read_csv(RootPath+DataFilename, sep=';') 
X = data_class[['area', 'grd_SD', 'grd_Quant']] # data without class
y = data_class[['class']] # class only

#%% Train model

# Generate random training dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.5, random_state=1) 

# Determine KNeighborsClassifier hyperparameters
krange = np.arange(1, 20, dtype=int)
erreurs = np.zeros(len(krange), dtype=float)
i=0
for k in krange:
    clf = nn.KNeighborsClassifier(k) 
    clf.fit(Xtrain,ytrain)
    erreurs[i]=1-clf.score(Xtest,ytest)   
    i=i+1
plt.plot(krange,erreurs)



# Train KNeighborsClassifier
clf = nn.KNeighborsClassifier(6) 
clf.fit(Xtrain,ytrain)







