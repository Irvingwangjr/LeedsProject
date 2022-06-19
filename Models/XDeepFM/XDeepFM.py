from typing import List

from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from Models.Layers.CINLayers import CINLayer
from Models.Layers.ConcatMlpLayers import ConcatMlpLayerV2
from Models.BaseModel import BaseModel
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.SeqTransform import GRULinearDimsTransModule
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2, DefaultMatchModule
from Models.Modules.FusionModule import XDeepFMFusion
import torch


class XDeepFM(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.layersDim = HyperParam.AutoIntMatchMlpDims
        self.featureNumb = HyperParam.AutoIntFeatureNum
        self.featureDim =  HyperParam.AutoIntFeatureDim
        self.CIN = CINLayer(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)
        self.DNN = ConcatMlpLayerV2( HyperParam.AutoIntMatchMlpDims)
        self.output = None


    def mainForward(self, feature):
        cin = self.CIN(feature)
        dnn = self.DNN(feature)
        # print(cat.shape)
        self.output = torch.sigmoid(dnn + cin)
        return self.output
    # ## sideInfo:dense/:处理纬度
    # def loadContextTransModule(self):
    #     return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)
    #
    # ## Transformer
    # def loadFusionModule(self):
    #     return XDeepFMFusion(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim, HyperParam.AutoIntMatchMlpDims)

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
    #     return DefaultMatchModule()
