from typing import List
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, AllConcatMlpWithLinear, AllConcatMlpWithWeightLinear
from Models.Modules.FusionModule import MultiLayerTransformer, MultiLayerTransformerWithField, FieldAwarePruningV2


class AutoPruningV2(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    def loadFusionModule(self):
        return FieldAwarePruningV2(HyperParam.AutoIntFeatureNum, HyperParam.AutoPruningFeatureDim,
                                   HyperParam.AutoPruningFieldDim, HyperParam.AutoPruningBeta,
                                   HyperParam.AutoPruningZeta,
                                   HyperParam.AutoPruningGamma)

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

    def endPoint(self):
        super().endPoint()
        featureL0 = HyperParam.AutoPruningFeatureL0 * self.fusionModule.featureL0
        # structureL0 = HyperParam.AutoPruningStructureL0 * self.fusionModule.structureL0
        interactionL0 = HyperParam.AutoPruningInteractionL0 * self.fusionModule.interactionL0
        loss = featureL0 + interactionL0
        self.addAuxiliaryLoss(loss)
