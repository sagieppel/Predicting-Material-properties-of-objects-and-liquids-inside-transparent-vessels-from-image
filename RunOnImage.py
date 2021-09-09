# Evaluate model predictio accuracy
# ...............................Imports..................................................................
import os
import numpy as np
import NetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import Reader as DataReader
import Visuallization as vis
import cv2
##################################Input paramaters#########################################################################################
# .................................Main Input parametrs...........................................................................................
InputImage = r"Examples//Image.jpg" # Input image path
InputMask = r"Examples//Mask.png" # Input mask path
Trained_model_path="logs_VesselMask//Defult.torch" # Trained model path
UseGPU=False # Run Net on GPU
####################Properties to  predict############################################################################################

OrderedPropertiesToPredict=["Transmission","Base Color","Metalic","Transmission Roguhness","Roughness"]  #List of property to predict in the right order Dictionary is orderless
DimPropertiesToPredict={'Transmission':1,'Base Color':3,'Metalic':1,'Transmission Roguhness':1,'Roughness':1} # Length of vector that reresent each property
ObjectsToPredict=["ContentMaterial","VesselMaterial"] # objects to predict

# =========================Load net weights====================================================================================================================

Net=NetModel.Net(OrderedObjectsToPredict=ObjectsToPredict,OrderedPropertiesToPredict=OrderedPropertiesToPredict,DimPropertiesToPredict=DimPropertiesToPredict) # Create net and load pretrained

Net.load_state_dict(torch.load(Trained_model_path))
Net = Net.cuda()#.eval()

#==============Load image and mask===========================================================
Im=cv2.imread(InputImage)
Msk=cv2.imread(InputMask)

if Msk.ndim>2: Msk=Msk.sum(2) # Convert to single channel
Msk[Msk>0]=1 # Convert mask  to 0/1 format

Im=np.expand_dims(Im,axis=0)
Msk=np.expand_dims(Msk,axis=0)
#===========Run net============================================================================
with torch.no_grad():
    Prd = Net.forward(Images=Im,ROIMask=Msk, UseGPU=UseGPU, TrainMode=False, FreezeBatchNorm_EvalON=True)  # Run net inference and get prediction

####*************************************Calculate satatistics*************************************************************************************************************************


txt=""
for nm in Prd:
    txt+=nm+"  \n "
    for ky in Prd[nm]:
        txt+=nm+"  ky: "
        PrdProperty=Prd[nm][ky][0].cpu().detach().numpy()
        txt += nm + " " +ky+ " : "+ str(PrdProperty)+"\n"
print(txt.replace('Base Color',"COLOR (RGB)"))

vis.show(np.hstack([Im[0],vis.GreyScaleToRGB(Msk[0]*200)]), "Input image and mask, see consule for predicted properties")
