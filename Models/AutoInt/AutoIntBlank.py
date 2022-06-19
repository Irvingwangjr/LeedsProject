from typing import List

from Models.BaseModelV2 import BaseModelV2
from Models.BaseModel import BaseModel
from Models.Modules.SeqTransform import BaseEmbedFormat
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, AllConcatMlpWithWeightLinear


class AutoIntBlank(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.output = None
        HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * HyperParam.AutoIntFeatureDim
        self.mlp = AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)

    def mainForward(self, feature):
        result = self.mlp(None, None, None, None, None, None, feature)
        return result

    # ## sideInfo:dense/:处理纬度
    # def loadContextTransModule(self):
    #     return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)
    #
    # ## Transformer
    # def loadFusionModule(self):
    #     return MultiLayerTransformerBlank(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
    #                                       HyperParam.AutoIntHeadNumList)
    #
    # ## None纬度变换
    # def loadItemTransModule(self):
    #     return LinearDimsTransModule(HyperParam.ItemFeatValVecFeatureDims, HyperParam.AutoIntFeatureDim, )
    #
    # # None：纬度变换
    # def loadUserTransModule(self):
    #     return GRULinearDimsTransModule(HyperParam.UserFeatValVecFeatureDims,
    #                                     HyperParam.AutoIntFeatureDim,
    #                                     HyperParam.AutoIntFeatureDim * len(self.nameList), self.nameList,
    #                                     HyperParam.SequenceLenKey)
    #
    # ##concatMLP
    # def loadMatchingModule(self):
    #     HyperParam.AutoIntWithFieldMatchMlpDims[0] = HyperParam.AutoIntFeatureNum
    #     return AllConcatMlpWithWeightLinear(HyperParam.AutoIntWithFieldMatchMlpDims, HyperParam.AutoIntFeatureNum,
    #                                         self.tempEmbed)
