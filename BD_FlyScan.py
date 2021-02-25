#%% Initialize
import cv2
import math
import napari
import joblib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed  

from skimage import io
from skimage.util import invert
from skimage.measure import regionprops
from skimage.filters.rank import gradient
from skimage.segmentation import clear_border
from skimage.draw import rectangle_perimeter, circle_perimeter
from skimage.morphology import disk, dilation, erosion, remove_small_objects, label

from sklearn.preprocessing import StandardScaler

#%% varnames
ROOTPATH = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey/21-02-05_TestMovies/'
FILENAME = 'Old_Pi05_BrightAdjusted.tif'

# General options
THRESH_COEFF = 5 # adjust auto thresholding (the smaller the more sensitive segmentation)

# display options
SHOW_BINARY_OUTLINES = 0
SHOW_ROI_OUTLINES = 1

# Advanced options
BIN_MIN_SIZE = 150 # remove binary objects smaller than BinMinSize1

#%% Subtract static background

def subtract_static_bg(stack):   
    '''Enter function general description + arguments'''
    mean_grey_val = np.mean(stack,0) # Get mean grey value
    bg_sub = stack - mean_grey_val # Subtract mean grey value from stack
    bg_sub = invert(bg_sub) # Invert stack  
    thresh = np.mean(bg_sub)
    
    return bg_sub, thresh

#%% Get binary mask

def get_binary_mask(bg_sub, thresh, THRESH_COEFF):
    '''Enter function general description + arguments'''
    binary_mask = bg_sub > thresh*THRESH_COEFF
    binary_mask = np.array(binary_mask, dtype=bool) 
    binary_mask = remove_small_objects(binary_mask, min_size=BIN_MIN_SIZE)
    binary_mask = dilation(binary_mask)
    binary_mask = clear_border(binary_mask)
    binary_mask = np.array(binary_mask, dtype=float)
    labels = label(binary_mask, connectivity=1)
    gradients = gradient(np.clip(bg_sub, 0, 255).astype('uint8'),disk(1)) 
    
    return binary_mask, labels, gradients

#%% Define regions of interest

def define_roi(roi_mask):
    '''Enter function general description + arguments'''    
    # comment
    roi_labels = label(roi_mask,connectivity=1)
    for prop in regionprops(roi_labels):
        if prop.eccentricity < 0.1:
            roi_labels[roi_labels == prop.label] = 3             
    # comment
    y_limits = np.mean(roi_mask,axis=1)    
    for i in range(nY):
        if  y_limits[i] == 0:
            roi_labels[i,:] = 3     
    # comment     
    roi_valid = np.ones([nY,nX])
    roi_valid[roi_labels==3] = 0
    
    return roi_labels, roi_valid

#%% Get objects properties

def get_obj_prop(labels, gradients):
    '''Enter function general description + arguments'''
    props = pd.DataFrame(columns=['timepoint', 'label', 'loc', 'area', 'ctrd_X','ctrd_Y', 'grd_SD', 'circularity'])   
    for i in range(nT):
        temp_labels = labels[i,:,:]
        temp_gradients = gradients[i,:,:,]
        for temp_props in regionprops(temp_labels):
            temp_mask = (temp_labels == temp_props.label).astype(bool)
            if np.mean(roi_valid[temp_mask]) >= 0.90:
                props = props.append({
                    'timepoint': i,
                    'label': temp_props.label,
                    'loc': roi_labels[round(temp_props.centroid[0]),round(temp_props.centroid[1])],
                    'area': temp_props.area,
                    'ctrd_X': temp_props.centroid[1],
                    'ctrd_Y': temp_props.centroid[0],
                    'grd_SD': np.std(temp_gradients[temp_mask]),
                    'circularity': (4 * math.pi * temp_props.area)/(temp_props.perimeter ** 2)}  
                    ,ignore_index=True)   
                
    return props

#%% Get objects predicted class (NB classifier)

def get_obj_class():    
    # Extract and format data
    X = props[['area', 'grd_SD', 'circularity']] # props
    X = X.to_numpy(float)
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Apply naive bayes classifier
    clf2 = joblib.load(ROOTPATH+'BD_FlyScan_Model.pkl')
    props_predict = pd.DataFrame(clf2.predict(X), columns=['class'])
    
    return props_predict

#%% Make a display


#%% Execute

# Open Stack from FILENAME
stack = io.imread(ROOTPATH+FILENAME)
roi_mask = io.imread(ROOTPATH+FILENAME[0:-4]+'_ROIMask.tif')/255
nT = stack.shape[0] # Get Stack dimension (t)
nY = stack.shape[1] # Get Stack dimension (x)
nX = stack.shape[2] # Get Stack dimension (y)

# Subtract static background
bg_sub, thresh = subtract_static_bg(stack)

# Get flies binary segmentation 
output_list = Parallel(n_jobs=35)(delayed(get_binary_mask)(bg_sub[i,:,:], thresh, THRESH_COEFF) for i in range(nT))
binary_mask = np.stack([arrays[0] for arrays in output_list], axis=0)
labels = np.stack([arrays[1] for arrays in output_list], axis=0)
gradients = np.stack([arrays[2] for arrays in output_list], axis=0)

# Define regions of interest
roi_labels, roi_valid = define_roi(roi_mask)

# Get objects properties
props = get_obj_prop(labels, gradients)

# Get objects class (NB classifier)
props_predict = get_obj_class()

# Save props as .csv
pd.DataFrame.to_csv(props,ROOTPATH+FILENAME[0:-4]+'_props.csv',index=False)
pd.DataFrame.to_csv(props_predict,ROOTPATH+FILENAME[0:-4]+'_props_predict.csv',index=False)

#%% temp


# Make tracking display
tracking_display = np.zeros([nT,nY,nX])
for i in range(len(props)):
    T = props['timepoint'][i].astype(int)
    ctrd_X = props['ctrd_X'][i].astype(int)
    ctrd_Y = props['ctrd_Y'][i].astype(int)
    
    # Draw detection squares
    rr, cc = rectangle_perimeter((ctrd_Y-50,ctrd_X-50), (ctrd_Y+50,ctrd_X+50), shape=tracking_display[T,:,:].shape)
    tracking_display[T,:,:][rr, cc] = 0.5  
    
    # Draw text
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    if ctrd_X <= nX-50: temp_X = ctrd_X+16
    else: temp_X = ctrd_X-46        
    if ctrd_Y <= nY-50: temp_Y = ctrd_Y+45
    else: temp_Y = ctrd_Y-36   
    tracking_display[T,:,:] = cv2.putText(tracking_display[T,:,:],str("{:03d}".format(i)),(temp_X,temp_Y), font, 0.75, (1,1,1), 1, cv2.LINE_AA)      
    
    # Draw valid circles
    if props_predict['class'][i] == 2:
        rr, cc = circle_perimeter(ctrd_Y,ctrd_X, 35, shape=tracking_display[T,:,:].shape)
        tracking_display[T,:,:][rr, cc] = 1 
        
# Make binary outlines
binary_outlines = np.zeros([nT,nY,nX])
for i in range(nT):
    binary_outlines[i,:,:,] = np.subtract(binary_mask[i,:,:,],erosion(binary_mask[i,:,:,],disk(1)))
    
# Make ROI outlines    
    roi_outlines = np.subtract(roi_valid,erosion(roi_valid,disk(1)))
 
# Merge layers    
if SHOW_BINARY_OUTLINES == 1 and SHOW_ROI_OUTLINES == 1:
    temp_display = np.maximum(tracking_display,binary_outlines)
    temp_display = np.maximum(temp_display,roi_outlines)
elif SHOW_BINARY_OUTLINES == 1 and SHOW_ROI_OUTLINES == 0:
    temp_display = np.maximum(tracking_display,binary_outlines)
elif SHOW_BINARY_OUTLINES == 0 and SHOW_ROI_OUTLINES == 1: 
    temp_display = np.maximum(tracking_display,roi_outlines)
elif SHOW_BINARY_OUTLINES == 0 and SHOW_ROI_OUTLINES == 0: 
    temp_display = tracking_display
temp_display = temp_display.astype('uint8')*255
display = np.maximum(stack,temp_display)    
    
io.imsave(ROOTPATH+FILENAME[0:-4]+'_Display.tif', display, check_contrast=True)
     
with napari.gui_qt():
    viewer = napari.view_image(display)