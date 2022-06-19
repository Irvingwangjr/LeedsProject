from typing import List
import torch
import torch.nn as nn
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, AllConcatMlpWithWeightLinear
from Models.Modules.FusionModule import LambdaGateFusionV3, LambdaGateFusionV4, LambdaGateFusionV5, LambdaGateFusionV6


# match换成DNN
# 实验发现，深度在3/4层效果不错，Head提升没有收益
class LambdaGateNetV6(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]

        super().__init__(featureInfo, device)
        xList = []
        yList = []
        for i in range(HyperParam.AutoIntFeatureNum):
            for j in range(i + 1, HyperParam.AutoIntFeatureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    def loadFusionModule(self):
        self.fieldWeight = nn.Parameter(torch.zeros(size=(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)))
        torch.nn.init.normal_(self.fieldWeight.data, mean=0, std=0.01)
        self.fieldGate = nn.Parameter(torch.zeros(size=(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)))
        torch.nn.init.normal_(self.fieldGate.data, mean=0, std=0.01)
        return LambdaGateFusionV6(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim,
                                  HyperParam.AutoIntFeatureDim,
                                  fieldWeight=self.fieldWeight, fieldGate=self.fieldGate)

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
        HyperParam.AutoIntMatchMlpDims[0] = int((HyperParam.AutoIntFeatureNum * (
                HyperParam.AutoIntFeatureNum - 1) / 2) * HyperParam.AutoIntFeatureDim)
        return AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims, hooker=self.hooker)

    def hooker(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
               contextTrans, fusionFeature):
        output = fusionFeature[:, self.x, self.y, :]
        return output
