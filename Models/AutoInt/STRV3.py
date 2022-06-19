from typing import List
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, AllConcatMlpWithLinear
from Models.Modules.FusionModule import MultiLayerTransformerSTRELUX


class STRV3(BaseModel):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.endEpoch = HyperParam.STRStartEpoch
        self.startEpoch = HyperParam.STREndEpoch
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## Transformer
    def loadFusionModule(self):
        return MultiLayerTransformerSTRELUX(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
                                            HyperParam.AutoIntHeadNumList, featureDim=HyperParam.AutoIntFeatureDim)

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
        return AllConcatMlpWithLinear(HyperParam.AutoIntMatchMlpDims, HyperParam.AutoIntFeatureNum, self.tempEmbed)

    def calBeforeEachEpoch(self, epochNumb):
        super().calBeforeEachEpoch(epochNumb)
        if epochNumb < self.startEpoch:
            self.fusionModule.setMaskTraining(isTrain=False)
        elif epochNumb >= self.startEpoch:
            self.fusionModule.setMaskTraining(isTrain=True)
        elif epochNumb >= self.endEpoch:
            self.fusionModule.setMaskTraining(isTrain=False)

    def buildParameterGroup(self):
        re = [
            {'params': self.contextTransModule.parameters()},
            {'params': self.itemTransModule.parameters()},
            {'params': self.userTransModule.parameters()},
            {'params': self.matchingModule.parameters()},
            {'params': self.fusionModule.transformer[0].QKVKernel.parameters()},
            {'params': self.fusionModule.transformer[0].concatRes.parameters()},
            {'params': self.fusionModule.STR.parameters(), 'weight_decay': 0},
            {'params': self.fusionModule.transformer[0].interactionPrune, 'weight_decay': 0}
        ]
        return re
