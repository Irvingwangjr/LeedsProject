from collections import OrderedDict
from typing import List

from Models.Layers.MultiHeadTransformerLayer import MultiHeadTransformerLayer
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from torch import nn

from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformer


class AutoInt(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        self.headNum = HyperParam.AutoIntHeadNumList
        self.layerDims = HyperParam.AutoIntLayerDims
        self.transformer = nn.Sequential(OrderedDict([
            (f'transformer{i}',
             MultiHeadTransformerLayer(self.featureNum, self.layerDims[i] * self.headNum[i], self.layerDims[i + 1],
                                       self.headNum[i])) for
            i in
            range(len(self.layerDims) - 1)])
        )
        self.output = None
        self.mlp = AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)

    def mainForward(self, feature):
        self.output = self.transformer(feature)
        result = self.mlp(None, None, None, None, None, None, self.output)
        return result

    # ## sideInfo:dense/:处理纬度
    # def loadContextTransModule(self):
    #     return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)
    #
    # ## Transformer
    # def loadFusionModule(self):
    #     return MultiLayerTransformer(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
    #                                  HyperParam.AutoIntHeadNumList)

    # ## None纬度变换
    # def loadItemTransModule(self):
    #     return LinearDimsTransModule(HyperParam.ItemFeatValVecFeatureDims, HyperParam.AutoIntFeatureDim, )
    #
    # # None：纬度变换
    # def loadUserTransModule(self):
    #     return GRULinearDimsTransModule(HyperParam.UserFeatValVecFeatureDims,
    #                                     HyperParam.AutoIntFeatureDim ,
    #                                     HyperParam.AutoIntFeatureDim* len(self.nameList), self.nameList,
    #                                     HyperParam.SequenceLenKey)
    #
    # ##concatMLP
    # def loadMatchingModule(self):
    #     return AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)
