from typing import List
import torch
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Models.Modules.MatchModule import AllConcatMlpWithWeightLinear
from Models.Modules.SeqTransform import GRULinearDimsTransModule

from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.FusionModule import LTEBilinearFusion


class LTEBiLinear(BaseModel):
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
        return LTEBilinearFusion(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
                                 HyperParam.AutoIntHeadNumList, HyperParam.AutoIntFeatureDim)

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
        HyperParam.AutoIntWithFieldMatchMlpDims[0] = int((HyperParam.AutoIntFeatureNum * (
                HyperParam.AutoIntFeatureNum - 1) / 2) * HyperParam.AutoIntFeatureDim)
        return AllConcatMlpWithWeightLinear(HyperParam.AutoIntWithFieldMatchMlpDims, HyperParam.AutoIntFeatureNum,
                                            self.tempEmbed, hooker=self.hooker)

    def hooker(self, userFeature, userTrans, itemFeature, itemTrans, contextFeature,
               contextTrans, fusionFeature):
        output = fusionFeature[:, self.x, self.y, :]
        return output
