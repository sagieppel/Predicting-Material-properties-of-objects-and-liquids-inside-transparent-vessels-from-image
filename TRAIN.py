# Train net Given an image of vessel and content and mask (region) of the vessel predict the material of the vessel content and the material of the vessel
#...............................Imports..................................................................
import os
import numpy as np
import NetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import Reader as DataReader
import Visuallization as vis
import cv2

#################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
DataFolder={} # Contain list of folders that will contain training data
DataFolder["LiquidContent"]=r"Examples/Train//"
# TransProteusFolder["ObjectContent"]=r"TranProteus/Training/ObjectContent/"
# TransProteusFolder["ObjectContent2"]=r"TranProteus/Training/SingleObjectContent/"
# TransProteusFolder["LiquidContent"]=r"TranProteus/Training/LiquidContent/"
# TransProteusFolder["LiquidFlat"]=r"TranProteus/Training/FlatSurfaceLiquids/

MinSize=280 # Min image dimension size (Height,width)
MaxSize=1000# Max image dimension size (Height,width)
MaxPixels=800*800*1.5# Max size of training batch in pixel, reduce to solve out of memory problems
MaxBatchSize=6#MAx number images in a batch
InputMaskType="VesselMask" ##"ContentMask" # Type of input mask for the net ("Vessel"/"Content")
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
Learning_Rate=1e-4 # learning rate
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
Weight_Decay=1e-5# Weight for the weight decay loss function
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
MAX_ITERATION = int(100000010) # Max  number of training iteration
##################################################################################################################################################
# ------------------------------ Properties to predict------------------------------------------------
OrderedPropertiesToPredict=["Transmission","Base Color","Metalic","Transmission Roguhness","Roughness"]  #List of property to predict in the right order Dictionary is orderless
DimPropertiesToPredict={'Transmission':1,'Base Color':3,'Metalic':1,'Transmission Roguhness':1,'Roughness':1} # List of peoperties  to predict and the vector size for each properry
ObjectsToPredict=["ContentMaterial","VesselMaterial"] # List of objects to predict properties of
#******************************Create folder for statics file and weights*********************************************************************************************************************

if not os.path.exists(TrainedModelWeightDir):
    os.mkdir(TrainedModelWeightDir)
#=========================Load net weights from previous run (if training was interupted)====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NetModel.Net(OrderedObjectsToPredict=ObjectsToPredict,OrderedPropertiesToPredict=OrderedPropertiesToPredict,DimPropertiesToPredict=DimPropertiesToPredict) # Create net and load pretrained

if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()

#------------------------------------Create optimizer-------------------------------------------------------------------------------------
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + "test" + ".torch")# Test saving the weight to see that all folders exists

#----------------------------------------# Create Array  of Readers for each input folder--------------------------------------------------------------------------------------------------------------

Readers={} # Array of Readers for each input folder
for nm in DataFolder:
    Readers[nm]=DataReader.Reader(DataFolder[nm],DimPropertiesToPredict,MaxBatchSize,MinSize,MaxSize,MaxPixels,TrainingMode=True)
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------

if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#-------------------Create statics dictionary  for keeping track of loss during training--------------------------------------------------------------------------------
PrevAvgLoss=0

AVGCatLoss={}
for nm in ObjectsToPredict:
   for ky in OrderedPropertiesToPredict:
        AVGCatLoss[nm+"_"+ky]=0
AVGCatLoss["Total"]=0
##############################################################################################################################
#..............Start Training loop: Main Training....................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop


    print("------------------------------",itr,"------------------------------------------------")
    readertype=list(Readers)[np.random.randint(len(list(Readers)))]  # Pick random reader (dataset)
    print(readertype)

    GTMaps, GTMaterials = Readers[readertype].LoadBatch() # Load training batch
    #################***************************************************************************************************
    # batchSize=GTMaps["VesselWithContentRGB"].shape[0]
    # for i in range(batchSize):
    #    for nm in GTMaps:
    #
    #          print(nm, GTMaps[nm][i].max(),GTMaps[nm][i].min())
    #          tmIm = GTMaps[nm][i].copy()
    #          if GTMaps[nm][i].max()>255 or GTMaps[nm][i].min()<0 or np.ndim(GTMaps[nm][i])==2:
    #              if tmIm.max()>tmIm.min():
    #                  tmIm[tmIm>1000]=0
    #                  tmIm = tmIm-tmIm.min()
    #                  tmIm = tmIm/tmIm.max()*255
    #              print(nm,"New", tmIm.max(), tmIm.min())
    #              if np.ndim(tmIm)==2: tmIm=cv2.cvtColor(tmIm, cv2.COLOR_GRAY2BGR)
    #          vis.show(np.hstack([tmIm,GTMaps["VesselWithContentRGB"][i].astype(np.uint8)]) ,nm+ " Max=" + str(GTMaps[nm][i].max()) + " Min=" + str(GTMaps[nm][i].min()))
    # #############*************************Run net and get prediction**********************************************************************

    Prd = Net.forward(Images=GTMaps["VesselWithContentRGB"],ROIMask=GTMaps[InputMaskType]) # Run net inference and get prediction
    Net.zero_grad()
    print("Calculating loss ")
 # #**************************************Calculate Loss *************************************************************************************************************************
    TotalLoss=0 # Total loss for every object and property
    LossCat={} # Loss by class and object

    for nm in Prd: # Loss for every object
        for ky in Prd[nm]: # Loss for every property
            GTProperty=torch.autograd.Variable(torch.from_numpy(GTMaterials[nm][ky]).cuda(),requires_grad=False) # Convert GT property to pytorch
            GTPropertyExist=torch.autograd.Variable(torch.from_numpy(GTMaterials[nm][ky+"_Exist"]).cuda(), requires_grad=False) # List that define wether proprty exist for the case
           # LossCat[nm]=(torch.pow(GTProperty-Prd[nm],2).mean(1)*GTPropertyExist).mean()
            LossCat[nm+"_"+ky] = (torch.abs(GTProperty - Prd[nm][ky]).mean(1) * GTPropertyExist).mean() # L1  loss
     #       if ky=='Base Color': print(nm," Prop exist", GTPropertyExist.mean())
            TotalLoss+=LossCat[nm+"_"+ky]
    LossCat["Total"]=TotalLoss

#---------------Total Loss  and running average loss----------------------------------------------------------------------------------------------------------
    print("Calculating Total Loss")
    fr = 1 / np.min([itr - InitStep + 1, 2000])

    for nm in LossCat:
        if not nm in AVGCatLoss: AVGCatLoss[nm]=0
        if  LossCat[nm]>0:
                AVGCatLoss[nm]=(1 - fr) * AVGCatLoss[nm] + fr * LossCat[nm].data.cpu().numpy() # Runnig average loss

#-----------------------Apply back propogation---------------------------------------------------------------------------------------------------

    TotalLoss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

############################################################################################################################3

    # Displat save and update learning rate

#########################################################################################################################
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 300 == 0:# Temprorary save model weight
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 60000 == 0 and itr>0: # permenantly save model weight
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")

#......................Write and display train loss statitics ..........................................................................

    if itr % 10==0:
        # Display train loss
        txt="\n"+str(itr)+"\tLearning Rate \t"+str(Learning_Rate)
        for nm in AVGCatLoss:
            txt+="\tAverage Cat Loss["+nm+"] "+str(AVGCatLoss[nm])+"  "
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
# #----------------Update learning rate -------------------------------------------------------------------------------
    if itr%10000==0:
        if "TotalPrevious" not in AVGCatLoss:
            AVGCatLoss["TotalPrevious"]=AVGCatLoss["Total"]
        elif AVGCatLoss["Total"]*0.95<AVGCatLoss["TotalPrevious"]: # If loss have decrease in less the 5% since last check, decrease learning rate
            Learning_Rate*=0.9
            if Learning_Rate<=4e-7: # If learning to small increase it back up
                Learning_Rate=5e-6
            print("Learning Rate="+str(Learning_Rate))
            print("======================================================================================================================")
            optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Update learning rate in optimizer
            torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
        AVGCatLoss["TotalPrevious"]=AVGCatLoss["Total"]+0.0000000001 # Save current average loss for future referance



