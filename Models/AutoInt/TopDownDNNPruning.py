from collections import OrderedDict
from typing import List

from Models.BaseTransforms.STRTransform import STRTransform
from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch import nn
import torch
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE


class headDNN(nn.Module):
    def __init__(self, head, inDim, outDim):
        super(headDNN, self).__init__()
        self.outDim = outDim
        self.inDim = inDim
        self.head = head
        self.para = nn.Parameter(torch.zeros(1, head, self.inDim, self.outDim))
        nn.init.normal(self.para.data, mean=0, std=0.01)

    def forward(self, feature):
        # print(feature.shape)
        re = torch.matmul(feature, self.para)
        act = torch.relu(re)
        return act


class TopDownDNNPruning(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        self.featureDim = HyperParam.AutoIntFeatureDim
        # self.init = -5
        # self.init = -10
        self.init = -150
        self.layerNorm = nn.LayerNorm([self.featureNum, self.featureDim])
        # self.topk = [30, 20, 20, 10, 10]
        # self.headNum = 50
        # self.headNum = 20
        self.headNum = 30
        # self.headNum = 40
        self.pruningWeight = nn.Parameter(self.init * torch.ones(self.featureNum, self.headNum))
        self.STR = STRTransform()
        self.classDNN = LinearTransform(
            layerDimension=[self.featureDim, self.headNum * 4, self.headNum * 2, self.headNum], dropLast=False)

        self.layerDims = HyperParam.AutoIntMatchMlpDims
        layer = []
        for i in range(len(self.layerDims) - 1):
            layer.append((f"headDNN:{i}", headDNN(self.headNum, self.layerDims[i], self.layerDims[i + 1])))
        self.featureDNN = nn.Sequential(OrderedDict(
            layer))
        self.out = LinearTransform([self.headNum, 16, 1], True)
        self.act = nn.Sigmoid()

    def mainForward(self, feature):
        feature = self.layerNorm(feature)
        cluster: torch.Tensor = self.classDNN(feature)
        # print(f"cluster dim {cluster.shape}")
        cluster: torch.Tensor = self.STR(cluster, self.pruningWeight)
        clusterInput = torch.transpose(cluster, 1, 2).unsqueeze(3) * feature.unsqueeze(1)
        # print(f"cclusterInput:size{clusterInput.shape}")
        cluster = clusterInput.reshape(-1, self.headNum, 1, self.featureNum * self.featureDim)
        fuse = self.featureDNN(cluster).squeeze()
        # print(fuse.shape)
        out = self.out(fuse)
        return self.act(out)
