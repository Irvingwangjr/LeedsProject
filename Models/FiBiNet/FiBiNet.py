import json
import os
from collections import OrderedDict
from typing import List

import torch
from Models.Layers.BiLinearLayer import BiLinearLayer
from Models.Layers.SeNetLayer import SeNetLayer

from DataSource.BaseDataFormat import *
from Models.BaseModelV2 import BaseModelV2
from torch import nn

from DataSource.WXBIZData import WXBIZData
from DataSource.WXBIZEmbed import WXBIZEmbed
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE, loadArgs, Config
from Models.Modules.MatchModule import AllConcatMlpV2, AllConcatMlpWithWeightLinear


class FiBiNet(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        xList = []
        yList = []
        for i in range(HyperParam.AutoIntFeatureNum):
            for j in range(i + 1, HyperParam.AutoIntFeatureNum):
                xList.append(i)
                yList.append(j)
        self.x = torch.Tensor(xList).type(torch.long)
        self.y = torch.Tensor(yList).type(torch.long)
        self.SeNet = SeNetLayer(HyperParam.AutoIntFeatureNum)
        self.BiLinear = BiLinearLayer(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)
        self.rawBiLinear = BiLinearLayer(HyperParam.AutoIntFeatureNum, HyperParam.AutoIntFeatureDim)
        HyperParam.AutoIntMatchMlpDims[0] = int((HyperParam.AutoIntFeatureNum * (
                HyperParam.AutoIntFeatureNum - 1) / 2) * HyperParam.AutoIntFeatureDim * 2)
        self.mlp = AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)

    def mainForward(self, feature):
        senet = self.SeNet(feature)
        biLinear = self.BiLinear(senet)
        rawBiLinear = self.rawBiLinear(feature)
        output = torch.cat([biLinear[:, self.x, self.y, :], rawBiLinear[:, self.x, self.y, :]], dim=1)
        result = self.mlp(None, None, None, None, None, None, output)
        return result


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    abs_address = '/Users/ivringwang/Desktop/tencent/GMM_torch/Config/FeatureConfig.json'
    featureInfo = loadArgs(abs_address)
    path = "/Users/ivringwang/Desktop/tencent/GMM_torch/test/dataset/train_tfdata/Data"
    test = WXBIZData(path)
    embed = WXBIZEmbed([], [], [], featureInfo)
    dataIter = test.getBatchData()

    with open(os.path.join(Config.absPath, Config.featureParamPath), 'r', encoding='utf-8') as feature:
        featureInfo: List[FeatureInfo] = json.load(feature, object_hook=FeatureInfo.hooker)

    model = FiBiNet(featureInfo, embed).to(device)
    optimizer = torch.optim.Adam(model.buildParameterGroup(), lr=HyperParam.LR, weight_decay=HyperParam.L2)
    loss_fn = eval(Config.lossType)().to(device)
    for i, count in dataIter:
        for j in i.keys():
            i[j] = torch.as_tensor(i[j]).to(device)
        result = model(i)
