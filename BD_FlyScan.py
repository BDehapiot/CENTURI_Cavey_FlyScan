#%% Initialize
import cv2
import napari
import numpy as np
import pandas as pd

from skimage import io
from skimage.util import invert
from skimage.measure import regionprops
from skimage.filters.rank import gradient
from skimage.draw import rectangle_perimeter
from skimage.segmentation import clear_border
from skimage.morphology import disk, dilation, erosion, remove_small_objects, label

#%% Inputs
RootPath = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey/21-02-05_TestMovies'
Filename = '/21-02-05_Pi04_DUP.tif'
ROIMaskname = '/Pi04_ROIMask.tif'

# General options
threshCoeff = 5 # adjust auto thresholding (the smaller the more sensitive segmentation)

# Display options
ShowBinaryOutlines = 0
ShowROIOutlines = 1

# Advanced options
BinMinSize1 = 150 # remove binary objects smaller than BinMinSize1

#%% Open stack

# Open Stack from filename
Stack = io.imread(RootPath+Filename)
ROIMask = io.imread(RootPath+ROIMaskname)/255
nT = Stack.shape[0] # Get Stack dimension (t)
nY = Stack.shape[1] # Get Stack dimension (x)
nX = Stack.shape[2] # Get Stack dimension (y)

#%% Subtract static background

BG = np.mean(Stack,0)
BGSub = Stack-BG 
BGSub = invert(BGSub)

# with napari.gui_qt():
#     viewer = napari.view_image(BGSub)
    
#%% Get BinaryMask (skimage) 

thresh = np.mean(BGSub)
def FlyScanBinaryMask(Input):
    # Thresholding
    BinaryMask = Input > thresh*threshCoeff
    # Remove small objects
    BinaryMask = np.array(BinaryMask, dtype=bool) # Convert to boolean
    BinaryMask = remove_small_objects(BinaryMask, min_size=BinMinSize1) # Remove small objects
    BinaryMask = dilation(BinaryMask) # Dilate Binary Mask
    BinaryMask = clear_border(BinaryMask) # Remove objects connected to the border
    BinaryMask = np.array(BinaryMask, dtype=float) # Convert back to float
    
    # Label objects
    Labels = label(BinaryMask, connectivity=1)
    # Get gradient of the image
    Gradient = gradient(np.clip(Input, 0, 255).astype('uint8'),disk(1))
    
    return BinaryMask, Labels, Gradient

# BinaryMask = np.zeros([nT,nY,nX])
# Labels = np.zeros([nT,nY,nX])
# Gradient = np.zeros([nT,nY,nX])
# for i in range(nT):
#     BinaryMask[i,:,:],Labels[i,:,:],Gradient[i,:,:] = FlyScanBinaryMask(BGSub[i,:,:])

# Parallel processing
from joblib import Parallel, delayed  
OutputList = Parallel(n_jobs=35)(delayed(FlyScanBinaryMask)(BGSub[i,:,:]) for i in range(nT))
SubList1 =[OutputList[0] for OutputList in OutputList]
SubList2 =[OutputList[1] for OutputList in OutputList]
SubList3 =[OutputList[2] for OutputList in OutputList]
BinaryMask = np.stack(SubList1, axis=0)
Labels = np.stack(SubList2, axis=0)
Gradient = np.stack(SubList3, axis=0)

# with napari.gui_qt():
#     viewer = napari.view_image(BinaryMask)

#%% Define Regions of interest

# Detect central circle
ROIMaskLabel = label(ROIMask,connectivity=1);
Unique = np.unique(ROIMaskLabel)
tempProps = regionprops(ROIMaskLabel)
for i in range(len(Unique)-1):
    if tempProps[i].eccentricity < 0.1:
        ROIMaskLabel[ROIMaskLabel == tempProps[i].label] = 2
    else:
        ROIMaskLabel[ROIMaskLabel == tempProps[i].label] = 1
                
# Get up and down horizontal limits
YLimits = np.mean(ROIMask,axis=1)
for i in range(nY):
    if YLimits[i] == 0:
        ROIMaskLabel[i,:] = 2
        
# Define ROIMaskValid        
ROIMaskValid = np.ones([nY,nX])
ROIMaskValid[ROIMaskLabel==2] = 0

#%% Get objects properties

MergedProps = pd.DataFrame(index=np.arange(0))
for i in range(nT):
    Unique = np.unique(Labels[i,:,:])
    rProps = regionprops(Labels[i,:,:])
    for j in range(len(Unique)-1):       
            tempProps = pd.DataFrame(index=np.arange(0),columns=['timepoint', 'label', 'area', 'ctrd_X','ctrd_Y', 'grd_SD', 'grd_Quant', 'class'])            
            tempCoords = rProps[j].coords
            tempValROIMaskValid  = np.zeros([len(tempCoords),1])
            tempValGradient = np.zeros([len(tempCoords),1])
            for k in range(len(tempCoords)-1):
                tempValROIMaskValid[k] = ROIMaskValid[tempCoords[k,0],tempCoords[k,1]]  
                tempValGradient[k] = Gradient[i,tempCoords[k,0],tempCoords[k,1]]                
            if np.mean(tempValROIMaskValid) >= 0.90:          
                tempProps.at[j,'timepoint'] = i
                tempProps.at[j,'label'] = rProps[j].label
                tempProps.at[j,'area'] = rProps[j].area
                tempProps.at[j,'ctrd_X'] = rProps[j].centroid[1]
                tempProps.at[j,'ctrd_Y'] = rProps[j].centroid[0]       
                tempProps.at[j,'grd_SD'] = np.std(tempValGradient)
                tempProps.at[j,'grd_Quant'] = np.quantile(tempValGradient,0.9)
                tempProps.at[j,'class'] = np.nan
                MergedProps = pd.concat([MergedProps, tempProps],ignore_index=True)     
    
# Normalize data
MergedProps['area'] = MergedProps['area']/np.mean(MergedProps['area'])
MergedProps['grd_SD'] = MergedProps['grd_SD']/np.mean(MergedProps['grd_SD'])
MergedProps['grd_Quant'] = MergedProps['grd_Quant']/np.mean(MergedProps['grd_Quant'])

#%% Make a Display

TrackedObjects = np.zeros([nT,nY,nX])
for i in range(len(MergedProps)):
    T = MergedProps['timepoint'][i]
    grd_SD = MergedProps['grd_SD'][i]
    grd_Quant = MergedProps['grd_Quant'][i]
    ctrd_X = MergedProps['ctrd_X'][i].astype(int)
    ctrd_Y = MergedProps['ctrd_Y'][i].astype(int)

    # Draw detection squares
    rr, cc = rectangle_perimeter((ctrd_Y-50,ctrd_X-50), (ctrd_Y+50,ctrd_X+50), shape=TrackedObjects[T,:,:].shape)
    TrackedObjects[T,:,:][rr, cc] = 1
    
    # Draw text
    if ctrd_X <= nX-50: tempX = ctrd_X+16
    else: tempX = ctrd_X-46        
    if ctrd_Y <= nY-50: tempY = ctrd_Y+45
    else: tempY = ctrd_Y-36
    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    TrackedObjects[T,:,:] = cv2.putText(TrackedObjects[T,:,:],str("{:03d}".format(i)),(tempX,tempY), font, 0.75, (1,1,1), 1, cv2.LINE_AA)      

# Merge layers
BinaryOutlines = np.zeros([nT,nY,nX])
for i in range(nT):
    BinaryOutlines[i,:,:,] = np.subtract(BinaryMask[i,:,:,],erosion(BinaryMask[i,:,:,],disk(1)))
ROIOutlines = np.subtract(ROIMask,erosion(ROIMask,disk(1)))

if ShowBinaryOutlines == 1 and ShowROIOutlines == 1:
    tempDisplay = np.maximum(TrackedObjects,BinaryOutlines)
    tempDisplay = np.maximum(tempDisplay,ROIOutlines)
elif ShowBinaryOutlines == 1 and ShowROIOutlines == 0:
    tempDisplay = np.maximum(TrackedObjects,BinaryOutlines)
elif ShowBinaryOutlines == 0 and ShowROIOutlines == 1: 
    tempDisplay = np.maximum(TrackedObjects,ROIOutlines)
elif ShowBinaryOutlines == 0 and ShowROIOutlines == 0: 
    tempDisplay = TrackedObjects
    
tempDisplay = tempDisplay.astype('uint8')*255
Display = np.maximum(Stack,tempDisplay)

# with napari.gui_qt():
#     viewer = napari.view_image(Display)
#     viewer = napari.view_image(ROIMaskValid)
#     viewer = napari.view_image(ROIMaskLabel)

#%% Save datas

MergedProps.to_csv(RootPath+Filename[0:-4]+'_MergedProps.csv')
io.imsave(RootPath+Filename[0:-4]+'_Gradient.tif', Gradient, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Display.tif', Display, check_contrast=True)