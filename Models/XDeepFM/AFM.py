import torch
import torch.nn as nn
from Models.Layers.ConcatMlpLayers import ConcatMlpLayerV2
from Models.BaseModelV2 import BaseModelV2, List
from Models.Modules.MatchModule import AllConcatMlpV2
from Utils.HyperParamLoadModule import HyperParam
from DataSource.BaseDataFormat import BaseEmbedFormat
from Utils.HyperParamLoadModule import FeatureInfo, FEATURETYPE


class AFM(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.neuralNumb = 1200
        self.featureNumb = HyperParam.AutoIntFeatureNum
        self.featureSize = HyperParam.AutoIntFeatureDim
        self.BNlog = nn.BatchNorm1d(self.featureNumb)
        self.BNExp = nn.BatchNorm1d(self.neuralNumb)
        self.weight = nn.Parameter(torch.zeros((1, self.neuralNumb, self.featureNumb, 1)))
        nn.init.normal(self.weight.data, mean=0, std=0.01)
        LayerDims = [self.neuralNumb * self.featureSize, 400, 400, 400, 1]
        self.Linear = ConcatMlpLayerV2( LayerDims)

    def mainForward(self, feature):
        log = feature
        log = self.BNlog(log)
        logTrans = self.weight * log.unsqueeze(1)
        interaction = torch.sum(logTrans, dim=-2)
        exp = torch.exp(interaction)
        bnexp = self.BNExp(exp)
        out = self.Linear(bnexp)
        result = torch.sigmoid(out)
        return result
