from collections import OrderedDict
from functools import reduce
from typing import List

from Models.BaseTransforms.LinearTransform import LinearTransform
from Models.Layers.ConcatMlpLayers import ConcatMlpLayerV2
from Models.Layers.SparseAttentionLayer import SparseDCNAttentionLayer, SENetMABLayer
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch import nn
import torch
from Models.Layers.InducedSetAttention import InducedSetAttention
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformer
from copy import deepcopy


class SparseSeNetInduceInt(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        self.depth = 4
        self.AuxLoss = 0.001
        self.LayerDims=[HyperParam.AutoIntFeatureNum,10,10,10]
        self.headNum = HyperParam.AutoIntHeadNumList
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.output = None
        self.induceAtte = nn.ModuleList(
            [SENetMABLayer(self.featureNum, self.featureDim) for i in range(self.depth)])
        dims = [self.depth * HyperParam.AutoIntFeatureDim, 16, 1]
        self.DNN = ConcatMlpLayerV2(dims)
        self.mlp = ConcatMlpLayerV2(HyperParam.AutoIntMatchMlpDims)
        # HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * sum(self.topk) * HyperParam.AutoIntFeatureDim
        self.interVec: torch.Tensor = None

    def mainForward(self, feature):
        setVec = feature
        result = []
        for i in range(self.depth):
            setVec, interaction = self.induceAtte[i](setVec)
            result.append(setVec)
            setVec = interaction
        out = torch.cat(result, dim=1)
        self.interVec = out
        mlp = self.mlp(feature)
        fuse = self.DNN(out)
        out = torch.sigmoid(mlp + fuse)
        # print(result.detach().numpy())
        return out

    # def getAuxLoss(self):
    #     # base = self.interVec.unsqueeze(2) - self.interVec.unsqueeze(1)
    #     # loss = torch.mean(torch.sqrt(torch.sum(base ** 2,dim=-1))) * self.AuxLoss
    #     # self.interVec=None
    #     return 0
