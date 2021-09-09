#
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Visuallization as vis
######################################################################################################################333
class Net(nn.Module):
########################################################################################################################
    def __init__(self, OrderedObjectsToPredict,OrderedPropertiesToPredict,DimPropertiesToPredict):
        # Generate net  base on resnext classifier with modified input and out put

        # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  Encoder (Resnext)----------------------------------------------------------


            self.Encoder = models.resnext101_32x8d(pretrained=True)


        # -----------------------------Input Mask Proccesing layer--------------------------------------------------------------------------------------------


            self.MaskEncoder = nn.Conv2d(1, 64, stride=2, kernel_size=3, padding=1, bias=True)


        # ----------------Final prediction layers------------------------------------------------------------------------------------------


            self.OutLayersList = nn.ModuleList()
            self.OutLayersDic = {}


            for nm in OrderedObjectsToPredict: # Add layer for each object and material property that need  be predict, this layer will use the last net layer as input
                    self.OutLayersDic[nm] = {}
                    for ky in OrderedPropertiesToPredict:
                        self.OutLayersDic[nm][ky] = nn.Linear(2048, DimPropertiesToPredict[ky],bias=False)
                        self.OutLayersList.append(self.OutLayersDic[nm][ky])

##########################################################################################################################################################
    def forward(self, Images, ROIMask,  UseGPU=True, TrainMode=True, FreezeBatchNorm_EvalON=False):

               #--------------------------Train mode take more precision then test mode--------------------------------------------
                if TrainMode == True:
                   tp = torch.FloatTensor
                else:
                   tp = torch.half
                   #      self.eval()
                   self.half()

                if FreezeBatchNorm_EvalON: self.eval() # Update batch norm statitics or freeze it
            #----------------------Convert inpot to pytorch----------------------------------------------------------------
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(np.float32)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)

                InROIMask = torch.autograd.Variable(torch.from_numpy(ROIMask.astype(np.float32)),requires_grad=False).unsqueeze(dim=1).type(tp)

    # ---------------Convert to cuda gpu or to CPU-------------------------------------------------------------------------------------------------------------------

                if UseGPU:
                    InpImages = InpImages.cuda()
                    InMask = InROIMask.cuda()
                    self.cuda()
                else:
                    InpImages = InpImages.cpu().float()
                    InMask = InROIMask.cpu().float()
                    self.cpu().float()
#----------------Normalize image values-----------------------------------------------------------------------------------------------------------

                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#---------------Run Encoder first layer-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#------------------Attention layer proccess input mask and add to the net  main features---------------------------------------------------------------------------
                x =x+ self.MaskEncoder(InMask)
#-------------------------Run remaining encoder layer------------------------------------------------------------------------------------------
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)

                x = self.Encoder.layer2(x)

                x = self.Encoder.layer3(x)

                x = self.Encoder.layer4(x)

                x = torch.mean(torch.mean(x, dim=2), dim=2)


                #===========================Run final prediction and predict properties=============================================================================================

                self.OutPredicts = {}

                for nm in  self.OutLayersDic:
                    self.OutPredicts[nm] = {}
                    for ky in self.OutLayersDic[nm]:
                            self.OutPredicts[nm][ky]=self.OutLayersDic[nm][ky](x)

                return self.OutPredicts










