from itertools import chain
from typing import List

import torch
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from torch import nn

from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformer, MLTestTransformerV2


class RETV2(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    def loadFusionModule(self):
        self.position = nn.Parameter(torch.zeros(size=(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)))
        nn.init.xavier_normal(self.position.data, gain=1.414)
        return MLTestTransformerV2(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim, 5, depth=2,
                                   position=self.position, hooker=self.hooker)

    ## None纬度变换
    def loadItemTransModule(self):
        return LinearDimsTransModule(HyperParam.ItemFeatValVecFeatureDims, HyperParam.AutoIntFeatureDim, )

    # None：纬度变换
    def loadUserTransModule(self):
        return GRULinearDimsTransModule(HyperParam.UserFeatValVecFeatureDims,
                                        HyperParam.AutoIntFeatureDim,
                                        HyperParam.AutoIntFeatureDim * len(self.nameList), self.nameList,
                                        HyperParam.SequenceLenKey)

    ##concatMLP
    def loadMatchingModule(self):
        return AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)

    def hooker(self, userTrans, userFeature, itemFeature, itemTrans, contextFeature, contextTrans):
        tempList = list(chain(userTrans, itemTrans, contextTrans))
        result=[]
        for feature in tempList:
            pass


