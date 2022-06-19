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
import torch.nn as nn


class CrossNet(nn.Module):
    def __init__(self, hiddenSize):
        super(CrossNet, self).__init__()
        self.hiddenSize = hiddenSize
        self.param = nn.Linear(hiddenSize, hiddenSize, bias=True)
        nn.init.normal(self.param.weight, std=0.01, mean=0)

    def forward(self, base, cross):
        result = base * (self.param(cross)) + cross
        return result


class DCNV2(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.depth = 2
        self.featureNumb = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.CrossNet = nn.ModuleList([CrossNet(self.featureDim * self.featureNumb) for i in range(self.depth)])
        self.DNN = ConcatMlpLayerV2(HyperParam.AutoIntMatchMlpDims)
        self.output = None

    def mainForward(self, feature):
        feature = torch.reshape(feature,(-1, self.featureDim * self.featureNumb))
        base = feature
        cross = feature
        for i in range(self.depth):
            cross = self.CrossNet[i](base, cross)
        out = torch.relu(cross)
        # print(cat.shape)
        dnn = self.DNN(out)
        self.output = torch.sigmoid(dnn)
        return self.output
