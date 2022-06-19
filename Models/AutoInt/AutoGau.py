from typing import List

from Models.AutoInt.AutoInt import AutoInt
from Models.Modules.FusionModule import MultiLayerTransformerGau
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from torch import nn
import torch
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam


class AutoGau(torch.autograd.Function):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        super(AutoGau, self).__init__(featureInfo, device)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    ## Gaussian
    def loadFusionModule(self):
        return MultiLayerTransformerGau(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
                                        HyperParam.AutoIntHeadNumList,)
    ## None纬度变换
    def loadItemTransModule(self):
        return LinearDimsTransModule(HyperParam.ItemFeatValVecFeatureDims, HyperParam.AutoIntFeatureDim, )

    # None：纬度变换
    def loadUserTransModule(self):
        return GRULinearDimsTransModule(HyperParam.UserFeatValVecFeatureDims,
                                        HyperParam.AutoIntFeatureDim ,
                                        HyperParam.AutoIntFeatureDim* len(self.nameList), self.nameList,
                                        HyperParam.SequenceLenKey)

    ##concatMLP
    def loadMatchingModule(self):
        return AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)
