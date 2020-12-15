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
from skimage.filters import threshold_triangle
from skimage.morphology import disk, dilation, remove_small_objects, label

#%% Inputs
RootPath = 'D:/CurrentTasks/CENTURIProject_IBDM_MatthieuCavey'
Filename = '/Chamber9(Dark)_(red)_RSize.tif'

# General options
threshCoeff = 2 # adjust auto thresholding (the smaller the more sensitive segmentation) (for test ml = 0.5)

# Advanced options
BinMinSize1 = 500 #

#%% Open stack

# Open Stack from filename
Stack = io.imread(RootPath+Filename)
nT = Stack.shape[0] # Get Stack dimension (t)
nY = Stack.shape[1] # Get Stack dimension (x)
nX = Stack.shape[2] # Get Stack dimension (y)

#%% Subtract static background

BG = np.mean(Stack,0)
BGSub = Stack-BG 
BGSub = invert(BGSub)

#%% Get BinaryMask (skimage) 

thresh = threshold_triangle(BGSub)
def FlyScanBinaryMask(Input):
    # Thresholding
    BinaryMask = Input > thresh*threshCoeff
    # Remove small objects
    BinaryMask = np.array(BinaryMask, dtype=bool) # Convert to boolean
    BinaryMask = remove_small_objects(BinaryMask, min_size=BinMinSize1)
    BinaryMask = dilation(BinaryMask)
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

#%% Get objects properties

MergedProps = pd.DataFrame(index=np.arange(0))
for i in range(nT):
    Unique = np.unique(Labels[i,:,:] )
    Props = pd.DataFrame(index=np.arange(len(Unique)-1),columns=['timepoint', 'label', 'area', 'ctrd_X','ctrd_Y', 'grd_SD', 'grd_Quant'])
    tempProps = regionprops(Labels[i,:,:])
    for j in range(len(Unique)-1):
        tempCoords = tempProps[j].coords
        tempValGradient = np.zeros([len(tempCoords),1])
        for k in range(len(tempCoords)-1):
            tempValGradient[k] = Gradient[i,tempCoords[k,0],tempCoords[k,1]]
        Props.at[j,'timepoint'] = i
        Props.at[j,'label'] = tempProps[j].label
        Props.at[j,'area'] = tempProps[j].area
        Props.at[j,'ctrd_X'] = tempProps[j].centroid[1]
        Props.at[j,'ctrd_Y'] = tempProps[j].centroid[0]       
        Props.at[j,'grd_SD'] = np.std(tempValGradient)
        Props.at[j,'grd_Quant'] = np.quantile(tempValGradient,0.9)
    MergedProps = pd.concat([MergedProps, Props],ignore_index=True) 
    
# Normalize data
MergedProps['area'] = MergedProps['area']/np.mean(MergedProps['area'])
MergedProps['grd_SD'] = MergedProps['grd_SD']/np.mean(MergedProps['grd_SD'])
MergedProps['grd_Quant'] = MergedProps['grd_Quant']/np.mean(MergedProps['grd_Quant'])

#%% Make a display

Display = np.zeros([nT,nY,nX])
for i in range(len(MergedProps)):
    T = MergedProps['timepoint'][i]
    grd_SD = MergedProps['grd_SD'][i]
    grd_Quant = MergedProps['grd_Quant'][i]
    ctrd_X = MergedProps['ctrd_X'][i].astype(int)
    ctrd_Y = MergedProps['ctrd_Y'][i].astype(int)

    # Draw detection squares
    rr, cc = rectangle_perimeter((ctrd_Y-50,ctrd_X-50), (ctrd_Y+50,ctrd_X+50), shape=Display[T,:,:].shape)
    Display[T,:,:][rr, cc] = 255 
    
    # Draw text
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    if grd_SD <= 1.1 and grd_Quant <= 1.1:
        Display[T,:,:] = cv2.putText(Display[T,:,:],'ceil',(ctrd_X-45,ctrd_Y-35), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif grd_SD > 1.1 and grd_Quant > 1.1:
        Display[T,:,:] = cv2.putText(Display[T,:,:],'floor',(ctrd_X-45,ctrd_Y-35), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

with napari.gui_qt():
    viewer = napari.view_image(Display)
    

#%% Open stack in napari

# with napari.gui_qt():
#     viewer = napari.view_image(temp)
    
#%% Save datas

# Change data type
BinaryMask = BinaryMask.astype('uint8')*255
Display = Display.astype('uint8')

# Saving
io.imsave(RootPath+Filename[0:-4]+'_BinaryMask.tif', BinaryMask, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Gradient.tif', Gradient, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Display.tif', Display, check_contrast=True)
