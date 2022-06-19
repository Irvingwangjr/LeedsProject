from typing import List

from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from Models.Layers.CINLayers import CINLayer
from Models.Layers.ConcatMlpLayers import ConcatMlpLayerV2
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
import torch


class FM(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        # self.layersDim = HyperParam.AutoIntMatchMlpDims
        # self.featureNumb = HyperParam.AutoIntFeatureNum
        # self.featureDim = HyperParam.AutoIntFeatureDim
        self.output = None

    def mainForward(self, feature: torch.Tensor):
        fm = torch.matmul(feature, torch.transpose(feature, 1, 2)).mean(dim=(1, 2)).unsqueeze(1)
        # print(cat.shape)
        self.output = torch.sigmoid(fm)
        return self.output
