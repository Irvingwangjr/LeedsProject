import os
from typing import List
import json
from abc import abstractmethod

import torch
from Models.Layers.ConcatGruLayers import ConcatGruLayer
from Models.BaseModel import BaseModel, FeatureInfo
from Models.Modules.DIms2DimTransModule import LinearItemTrans
from Models.Modules.SeqTransform import UserDefaultTrans
from Models.Modules.FusionModule import GaussianFusionModule

from Utils.HyperParamLoadModule import HyperParam, FEATURETYPE, Config


class GMM(BaseModel):

    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    def forward(self, userFeature, itemFeature, contextFeature):
        return super().forward(userFeature, itemFeature, contextFeature)

    def loadItemTransModule(self):
        return LinearItemTrans(self.tempEmbed, self.hyperParams.GMMBaseLinearItemTransLayerDims)

    def loadContextTransModule(self):
        return DefaultContextTrans()

    def loadUserTransModule(self):

        return ConcatGruLayer(self.tempEmbed * len(self.nameList), self.tempEmbed,
                              self.nameList, HyperParam.SequenceLenKey)

    def loadFusionModule(self):
        gaussianInputDim = [self.tempEmbed for i in
                            range(HyperParam.UserFeatIdxLookupFeatureNumb + 1)]
        for i in HyperParam.UserFeatValVecFeatureDims.keys():
            gaussianInputDim.append(HyperParam.UserFeatValVecFeatureDims[i])
        return GaussianFusionModule(1 + HyperParam.UserFeatIdxLookupFeatureNumb + HyperParam.UserFeatValVecFeatureNumb,
                                    gaussianInputDim,
                                    self.hyperParams.GMMV1GaussianFusionOutDims, backupHooker=self.GaussionFusionHooker)

    def GaussionFusionHooker(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature,
                             contextTrans):
        # 有可能有风险，依赖了字典 key 的顺序
        keyVal = []
        keyVal.append(userTrans)
        for i in userFeature.keys():
            if ("uin_vid" not in i):
                keyVal.append(userFeature[i])

        matchVal = [itemTrans for i in range(len(keyVal))]
        return keyVal, matchVal

