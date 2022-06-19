from typing import List

from Models.AutoInt.STRV3 import STRV3
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformerSTRX


class STRV4(STRV3):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(featureInfo, device)

    ## Transformer
    def loadFusionModule(self):
        return MultiLayerTransformerSTRX(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntLayerDims,
                                        HyperParam.AutoIntHeadNumList, featureDim=HyperParam.AutoIntFeatureDim)


    def buildParameterGroup(self):
        re = [
            {'params': self.contextTransModule.parameters()},
            {'params': self.itemTransModule.parameters()},
            {'params': self.userTransModule.parameters()},
            {'params': self.matchingModule.parameters()},
            {'params': self.fusionModule.transformer[0].QKVKernel.parameters()},
            {'params': self.fusionModule.transformer[0].concatRes.parameters()},
            {'params': self.fusionModule.transformer[0].interactionPrune, 'weight_decay': 0}
        ]
        return re

