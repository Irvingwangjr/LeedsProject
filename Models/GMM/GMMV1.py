import os
from typing import List
import torch
import json
from Models.GMM.GMM import GMM
from Models.Modules.FusionModule import GaussianFusionModule
from Utils.HyperParamLoadModule import FeatureInfo, FEATURETYPE, Config
from Models.Modules.MatchModule import *
from Utils.HyperParamLoadModule import HyperParam
from Models.Modules.SeqTransform import ConcatGruLayer


class GMMV1(GMM):

    def __init__(self, featureInfo: List[FeatureInfo],device):

        super().__init__(featureInfo,device)

    def forward(self, userFeature, itemFeature, contextFeature):
        return super().forward(userFeature, itemFeature, contextFeature)

    def loadFusionModule(self):
        gaussianInputDim = [self.tempEmbed for i in
                            range(HyperParam.UserFeatIdxLookupFeatureNumb + 1)]
        for i in HyperParam.UserFeatValVecFeatureDims.keys():
            gaussianInputDim.append(HyperParam.UserFeatValVecFeatureDims[i])
        return GaussianFusionModule(1 + HyperParam.UserFeatIdxLookupFeatureNumb + HyperParam.UserFeatValVecFeatureNumb,
                                    gaussianInputDim,
                                    self.hyperParams.GMMV1GaussianFusionOutDims, backupHooker=self.GaussionFusionHooker)


    def loadMatchingModule(self):
        # return UserConcatMlp(self.userFeatureNumb,self.hyperParams.UserConcatMlpDims)
        # return itemGaussian(self.userFeatureNumb,self.hyperParams.LinearItemTransLayerDims[-1],self.hyperParams.GaussianFusionOutDims)
        return MuItemAttention(hooker=self.matchingHooker)

    def matchingHooker(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
                       contextTrans, fusionFeature):
        fusion: GaussianFusionModule = self.fusionModule
        muTemp = [i.mu[None, :, :] for i in fusion.gaussian.gaussianKernel]
        mu = torch.cat(muTemp, 0).permute(dims=(1, 2, 0))
        matchScore = torch.cat(fusionFeature, 0).permute((1, 0, 2))
        return mu, itemTrans[:, None, :], matchScore

    def loadContextTransModule(self):
        return MapUserItemContextTrans(HyperParam.GMMV1ContextTransDims, HyperParam.GMMV1ContextTransDims)
