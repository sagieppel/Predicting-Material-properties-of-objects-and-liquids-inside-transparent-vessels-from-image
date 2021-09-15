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
TestFolder = r"Examples/Train/" # Test folder
#TestFolder = r"TranProteus/Testing/LiquidContent/" # Test folder
Trained_model_path="logs//Defult.torch" # Weights of model to test
InputMask="VesselMask"#"ContentMask" # Name of the mask that will be given as input to the vessel#"ContentMask", # VesselMask

####################Properties to  predict############################################################################################

OrderedPropertiesToPredict=["Transmission","Base Color","Metalic","Transmission Roguhness","Roughness"]  #List of property to predict in the right order Dictionary is orderless
DimPropertiesToPredict={'Transmission':1,'Base Color':3,'Metalic':1,'Transmission Roguhness':1,'Roughness':1} # Length of vector that reresent each property
ObjectsToPredict=["ContentMaterial","VesselMaterial"] # objects to predict

# =========================Load net weights====================================================================================================================

Net=NetModel.Net(OrderedObjectsToPredict=ObjectsToPredict,OrderedPropertiesToPredict=OrderedPropertiesToPredict,DimPropertiesToPredict=DimPropertiesToPredict) # Create net and load pretrained

Net.load_state_dict(torch.load(Trained_model_path))
Net = Net.cuda()#.eval()
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

Reader = DataReader.Reader(TestFolder, DimPropertiesToPredict, 1, 280, 1000, 800 * 800 * 2,TrainingMode=False)

# -------------------Cat losss--------------------------------------------------------------------------------
Statitics=0
SumAbsDiffrence = {}
SumSamples = {}
for nm in ObjectsToPredict:

    SumAbsDiffrence[nm] = {}
    SumSamples[nm] = {}
    for ky in OrderedPropertiesToPredict:
        SumAbsDiffrence[nm][ky] = 0
        SumSamples[nm][ky] = 0


# ..............Start Evaluation....................................................................
print("Start Evaluation")
while(Reader.epoch==0):  # Main training loop
#for i in range(50000):
    print("------------------------------", Reader.itr, "------------------------------------------------")

    GTMaps, GTMaterials = Reader.LoadSingle(1000) # Load single image and annotation
  #  if not 'Distribution' in  GTMaterials["ContentMaterial"]: continue
    ##### ***************************************************************************************************
    # if i > 1400:
    # for nm in GTMaterials:
    #     print(nm,"=",GTMaterials[nm])
    #     Img = GTMaps['VesselWithContentRGB'][0].copy()
    #     Im2=Img.copy()
    #     Im2[:,:,0][GTMaps['ContentMask'][0] > 0] = 255
    #     Im2[:, :, 1][GTMaps['VesselMask'][0] > 0] = 255
    #     vis.show(np.hstack([Img,Im2]),str(GTMaterials))
    #*****************************************************************************
    # print("RUN PREDICITION")
    with torch.no_grad():
        Prd = Net.forward(Images=GTMaps["VesselWithContentRGB"],ROIMask=GTMaps[InputMask],TrainMode=False)  # Run net inference and get prediction
    Net.zero_grad()
    # ####*************************************Calculate satatistics*************************************************************************************************************************


    for nm in Prd:
        if not (nm in GTMaterials): continue
        for ky in Prd[nm]:
            if not 'Distribution' in  GTMaterials["ContentMaterial"]: continue # Ignore none BSDF materials
         #   if GTMaterials[nm]["Transmission"]>0.8: continue
           # if (ky == "Metalic" or ky == "Transmission") and (GTMaterials[nm]["Roughness"]>0.2): continue # When roughness is high metallic and transmission are meaning less
            if not (ky in GTMaterials[nm]):continue # ignore properties that are not in the GT annotation
            GTProperty = GTMaterials[nm][ky]
            PrdProperty=Prd[nm][ky][0].cpu().detach().numpy()
            SumAbsDiffrence[nm][ky] += np.abs(GTProperty - PrdProperty).mean() # Calculate absolute difference between predict and GT property
            SumSamples[nm][ky] += 1

    ###########################################Display statics##################################################################################################
    print("###########################################", Reader.itr, "#############################################################")
    for nm in SumAbsDiffrence:
        print("----------------------------",nm,"------------------------------------------------")
        for ky in SumAbsDiffrence[nm]:
            if SumSamples[nm][ky]>0:
               print(nm,"\t",ky,"\tMAE=\t",str(SumAbsDiffrence[nm][ky]/SumSamples[nm][ky]),"\tNumber of Samples=\t",SumSamples[nm][ky])



