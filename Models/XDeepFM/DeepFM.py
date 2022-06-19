from typing import List

from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from Models.Layers.CINLayers import CINLayer
from Models.Layers.ConcatMlpLayers import ConcatMlpLayerV2
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
import torch


class DeepFM(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.layersDim = HyperParam.AutoIntMatchMlpDims
        self.featureNumb = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.DNN = ConcatMlpLayerV2(HyperParam.AutoIntMatchMlpDims)
        self.lr = torch.nn.Linear(2, 1, bias=True)
        torch.nn.init.normal(self.lr.weight.data, mean=0, std=0.01)
        self.output = None

    def mainForward(self, feature: torch.Tensor):
        dnn = self.DNN(feature)
        fm = torch.matmul(feature, torch.transpose(feature, 1, 2)).sum(dim=(1, 2)).unsqueeze(1)
        norm=fm/self.featureNumb**2
        # print(cat.shape)
        out = self.lr(torch.cat((norm, dnn), dim=1))
        self.output = torch.sigmoid(out)
        return self.output
