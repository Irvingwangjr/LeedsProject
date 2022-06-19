from typing import List
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, DefaultMatchModule
from Models.Modules.FusionModule import XDeepFMFusion, XDeepFMFusionV2, LinearBiLinearFusion
import torch
import copy


class LinearBiLinear(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    def loadFusionModule(self):
        temp1 = [10, 256, 128, 64]
        temp2 = [10, 256, 128, 64]
        temp1[0] = HyperParam.AutoIntMatchMlpDims[0]
        temp2[0] = int((HyperParam.AutoIntFeatureNum * (
                HyperParam.AutoIntFeatureNum - 1) / 2) * HyperParam.AutoIntFeatureDim)
        return LinearBiLinearFusion(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim,
                                    BiLayersDim=temp2, layerDim=temp1)

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
        return DefaultMatchModule()
