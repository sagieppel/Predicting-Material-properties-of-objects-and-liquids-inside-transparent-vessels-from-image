# Reader for transproteus material properties

import numpy as np
import os
import cv2
import json
import threading
import Visuallization as vis

MapsAndDepths= { # List of properties and object to load and their vector size
               "VesselMask":1,
               "ContentMask":1,
               "VesselWithContentRGB":3,
               "ContentMaterial":0,
               "VesselMaterial":0
}

#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"", PropertiesToPredict={},MaxBatchSize=1,MinSize=250,MaxSize=900,MaxPixels=800*800*5,TrainingMode=True):
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.PropertiesToPredict=PropertiesToPredict # Properties for the reader to extract and their shape
# ----------------------------------------Create list of annotations--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir): #
            AnnDir=MainDir+"//"+AnnDir+"//"
            Ent={}
            if not (os.path.isfile(AnnDir+"//ContentMaterial.json") and os.path.isfile(AnnDir + "//VesselMaterial.json")): continue
            Ent["ContentMaterial"]=AnnDir+"//ContentMaterial.json"
            Ent["VesselMaterial"] = AnnDir + "//VesselMaterial.json"
            Ent["VesselMask"] = AnnDir + "//VesselMask.png"
            Ent["MainDir"]=AnnDir
            for nm in os.listdir(AnnDir):
                filepath=AnnDir+"/"+nm
                if ("VesselWithContent" in nm) and ("_RGB.jpg" in nm):
                    EntTemp = Ent.copy()
                    EntTemp["VesselWithContentRGB"] = filepath
                    EntTemp["ContentMask"] = EntTemp["VesselWithContentRGB"].replace("_RGB.jpg", "_Mask.png").replace("VesselWithContent_", "Content_")
                    self.AnnList.append(EntTemp)
#------------------------------------------------------------------------------------------------------------

        print("done making file list Total=" + str(len(self.AnnList)))
        if TrainingMode:
            self.StartLoadBatch() # Start loading next batch in background (multithreaded)


#############################################################################################################################

# Crop and resize image and mask and ROI to fit batch size

#############################################################################################################################
        # Crop and resize image and mask and ROI to fit batch size
    def CropResize(self, Maps, Hb, Wb,AllowResize):
            # ========================resize image if it too small to the batch size==================================================================================

            h, w,d = Maps["VesselWithContentRGB"].shape
            Bs = np.min((h / Hb, w / Wb))
            if (
                    Bs < 1 or Bs > 3 or np.random.rand() < 0.2):  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
                h = int(h / Bs) + 1
                w = int(w / Bs) + 1
                for nm in Maps:
                    if hasattr(Maps[nm], "shape"):  # check if array
                        if "RGB" in nm:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                        else:
                            Maps[nm] = cv2.resize(Maps[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

            # =======================Crop image to fit batch size===================================================================================

            if w > Wb:
                X0 = np.random.randint(w - Wb)
            else:
                X0 = 0
            if h > Hb:
                Y0 = np.random.randint(h - Hb)
            else:
                Y0 = 0

            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    Maps[nm] = Maps[nm][Y0:Y0 + Hb, X0:X0 + Wb]

            # -------------------If still not batch size resize again-------------------------------
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):  # check if array
                    if not (Maps[nm].shape[0] == Hb and Maps[nm].shape[1] == Wb):
                        Maps[nm] = cv2.resize(Maps[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            return Maps

#################################################Generate Annotaton mask###################################################################
######################################################Augmented Image##################################################################################################################################
    def Augment(self,Maps):

        if np.random.rand()<0.5: # flip left right
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):
                            Maps[nm]= np.fliplr(Maps[nm])

        if np.random.rand() < 0.2:  # rotate 90
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):
                      Maps[nm] = np.rot90(Maps[nm])
        if np.random.rand() < 0.2:  # flip up down
            for nm in Maps:
                if hasattr(Maps[nm], "shape"):
                     Maps[nm] = np.flipud(Maps[nm])

        return Maps
############################################################################################################################################################
##################################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, pos, Hb, Wb):
# -----------------------------------Select  random image-----------------------------------------------------------------------------------------------------
            AnnInd = np.random.randint(len(self.AnnList))
            Ann=self.AnnList[AnnInd]
#------------------open dicionary file-------------------------------------------
            PropDic = {}
            with open(Ann["ContentMaterial"]) as f:
                PropDic["ContentMaterial"] = json.load(f)
            with open(Ann["VesselMaterial"]) as f:
                PropDic["VesselMaterial"] = json.load(f)
#--------------------------'TransparentLiquidMaterial' is special case where some properties are missing and need to be completed--------------------
            if  PropDic["VesselMaterial"]['Name']== 'TransparentLiquidMaterial':
                    PropDic["VesselMaterial"]['Transmission']=[0.97]
                    PropDic["VesselMaterial"]['Metalic'] = [0]


            if PropDic["ContentMaterial"]['Name'] == 'TransparentLiquidMaterial':
                    PropDic["ContentMaterial"]['Transmission'] = [0.97]
                    PropDic["ContentMaterial"]['Metalic'] = [0]

#-----------------------------------Load image and segmentation mask---------------------------------------------------------------
            Maps={}

            for nm in MapsAndDepths:
                if (not nm in Ann) or MapsAndDepths[nm]==0: continue
                Path = Ann[nm]
                I = cv2.imread(Path)

                Maps[nm] = I.astype(np.float32)

            Maps["VesselMask"]=(Maps["VesselMask"].sum(2)>0).astype(np.float32) #  Convert mask to 0/1 format
            Maps["ContentMask"]=(Maps["ContentMask"].sum(2)>0).astype(np.float32) #  Convert mask to 0/1 format






#-----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
            Maps = self.Augment(Maps)
            if Hb!=-1:
               Maps = self.CropResize(Maps, Hb, Wb,AllowResize=False)

#----------------------Generate forward and background segment mask-----------------------------------------------------------------------------------------------------------
            if Maps["ContentMask"].mean() < 0.005 or Maps["VesselMask"].mean() < 0.005: # If there no vessel or content read other image (could be cut away in the cropping stage)
                return self.LoadNext(pos, Hb, Wb)
  #----------------------Add loaded data into the training batch-----------------------------------------------------
            for nm in Maps: # Add image and mask to batch
             #   if nm in  self.Maps:
                     self.Maps[nm][pos]=Maps[nm]

            for nm in PropDic: # Add properties to batch
                    for ky in self.PropertiesToPredict:
                        if not ky in PropDic[nm]: continue
                        PropertySize=self.PropertiesToPredict[ky]
                        self.Properties[nm][ky + "_Exist"][pos] = 1


                        if PropertySize > 1:
                            self.Properties[nm][ky][pos] = np.array(PropDic[nm][ky])[:PropertySize]
                        else:
                            self.Properties[nm][ky][pos] = np.array([PropDic[nm][ky]])[:PropertySize]
            return True
##################################################################################################################################################################

#         Start load next batch

############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb =np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight #900
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width #900
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        #====================Creating empty batch===========================================================
        self.Maps={}
        self.Properties={}
        for nm in MapsAndDepths:
            if MapsAndDepths[nm]>1:
                self.Maps[nm]= np.zeros([BatchSize, Hb, Wb,MapsAndDepths[nm]], dtype=np.float32)
            elif MapsAndDepths[nm]==1:
                self.Maps[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
            else:
                self.Properties[nm] = {}
                for  ky in self.PropertiesToPredict:
                       self.Properties[nm][ky]=np.zeros([BatchSize,  self.PropertiesToPredict[ky]], dtype=np.float32)
                       self.Properties[nm][ky+"_Exist"] = np.zeros([BatchSize], dtype=np.float32) # Does property exist in dictionary
        #===================Start loading data (multi threaded)===================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="threadReader"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
            self.WaitLoadBatch() # whait for
            Maps=self.Maps
            Properties=self.Properties
#.......................................................................................................................
            self.StartLoadBatch() # Start loading next batch
            return Maps, Properties

##################################################################################################################################################################
# ==========================Read image annotation and data for single image without augmentation===============================================================================================
    def LoadSingle(self, MaxSize):
# -----------------------------------pick next image-----------------------------------------------------------------------------------------------------
            if self.itr >= len(self.AnnList):
                self.itr = 0
                self.epoch += 1

            Ann = self.AnnList[self.itr]
            self.itr += 1
#---------------------------Load dictionary from json files------------------------------------------------------------------
            PropDic = {}
            with open(Ann["ContentMaterial"]) as f:
                PropDic["ContentMaterial"] = json.load(f)
            with open(Ann["VesselMaterial"]) as f:
                PropDic["VesselMaterial"] = json.load(f)
#--------------------------'TransparentLiquidMaterial' is special case where some properties are missing and need to be completed (not bsdf materials)--------------------
            if  PropDic["VesselMaterial"]['Name']== 'TransparentLiquidMaterial':
                    PropDic["VesselMaterial"]['Transmission']=[0.97]
                    PropDic["VesselMaterial"]['Metalic'] = [0]
            if PropDic["ContentMaterial"]['Name'] == 'TransparentLiquidMaterial':
                    PropDic["ContentMaterial"]['Transmission'] = [0.97]
                    PropDic["ContentMaterial"]['Metalic'] = [0]
#-----------------------------------Load masks and image---------------------------------------------------------------
            Maps={}

            for nm in MapsAndDepths:
                if (not nm in Ann) or MapsAndDepths[nm]==0: continue
                Path = Ann[nm]
                I = cv2.imread(Path)

                Maps[nm] = I.astype(np.float32)
            Maps["VesselMask"]=(Maps["VesselMask"].sum(2)>0).astype(np.float32) # Convert to 0/1 format
            Maps["ContentMask"]=(Maps["ContentMask"].sum(2)>0).astype(np.float32) # Convert to 0/1 format



#-----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
            h, w = Maps["VesselMask"].shape
            r = np.min([MaxSize / h, MaxSize / w])
            if r < 1:
                for nm in Maps:
                    Maps[nm] = cv2.resize(Maps[nm], dsize=(int(r * w), (r * w)), interpolation=cv2.INTER_NEAREST)
            for nm in Maps: Maps[nm] = np.expand_dims(Maps[nm], axis=0)
          #
#----------------------Trasnform properties into proper size array -----------------------------------------------------------------------------------------------------------

            for nm in PropDic:
                    for ky in self.PropertiesToPredict:
                        if not ky in PropDic[nm]: continue
                        PropertySize=self.PropertiesToPredict[ky]
              #          PropDic[nm][ky][nm][ky + "_Exist"] = 1


                        if PropertySize > 1:
                            PropDic[nm][ky] = np.array(PropDic[nm][ky])[:PropertySize]
                        else:
                            PropDic[nm][ky] = np.array([PropDic[nm][ky]])[:PropertySize]


            return Maps, PropDic
