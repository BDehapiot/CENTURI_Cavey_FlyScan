#%% Initialize
import os
import joblib
import pandas as pd

from sklearn import naive_bayes as nb
from sklearn.preprocessing import StandardScaler

#%% Inputs
ROOTPATH = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey/21-02-05_TestMovies/'

#%% Get data

def get_data():
    '''Enter function general description + arguments'''
    dirlist = os.listdir(ROOTPATH)
    merged_props = pd.DataFrame(columns=['timepoint', 'label', 'area', 'ctrd_X','ctrd_Y', 'grd_SD', 'circularity'])
    merged_class = pd.DataFrame(columns=['class'])
    for temp_name in dirlist:
        if 'props' in temp_name:
            props_path = (ROOTPATH+temp_name)
            class_path = (ROOTPATH+temp_name[0:-4]+'_class.csv')
            if os.path.exists(class_path) == True:
                temp_props = pd.read_csv(props_path, sep=',') 
                temp_class = pd.read_csv(class_path, sep=',') 
                merged_props = pd.concat([merged_props, temp_props], ignore_index=True)   
                merged_class = pd.concat([merged_class, temp_class], ignore_index=True)
    
    return merged_props, merged_class

#%% Train model 

def train_model():
    '''Enter function general description + arguments'''
    # Extract and format data
    X = merged_props[['area', 'grd_SD', 'circularity']] # props
    y = merged_class[['class']] # class
    X = X.to_numpy(float)
    y = y.to_numpy(float)
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Train naive bayes classifier
    clf = nb.GaussianNB()  
    clf.fit(X, y)
    
    return clf

#%% Execute

# Get data
merged_props, merged_class = get_data()

# Train model 
clf = train_model()

# Save model
joblib.dump(clf, ROOTPATH+'BD_FlyScan_Model.pkl') 

#%% temp
