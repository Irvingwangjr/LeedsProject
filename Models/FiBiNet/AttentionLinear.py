from typing import List
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import SeNetBiLinearFusionV2, AttentionLinearFusion
import torch


class AttentionLinear(BaseModel):
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
        return AttentionLinearFusion(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)

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
