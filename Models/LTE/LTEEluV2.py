from typing import List

from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Models.Modules.MatchModule import AllConcatMlpWithWeightLinear
from Models.Modules.SeqTransform import GRULinearDimsTransModule

from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.FusionModule import MultiLayerTransformer, MultiLayerTransformerBlank, TransformerLTE, \
    TransformerLTEElu, TransformerLTE, TransformerLTEEluV2


class LTEEluV2(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)
    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)
    ## Transformer
    def loadFusionModule(self):
        return TransformerLTEEluV2(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
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
        return AllConcatMlpWithWeightLinear(HyperParam.AutoIntWithFieldMatchMlpDims, HyperParam.AutoIntFeatureNum,
                                            self.tempEmbed)
