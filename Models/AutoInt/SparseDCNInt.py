from collections import OrderedDict
from functools import reduce
from typing import List
from Models.Layers.SparseAttentionLayer import SparseDCNAttentionLayer
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch import nn
import torch
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformer


class SparseDCNInt(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        # self.topk = [60, 60, 40, 40,30]
        # self.topk=[100,80,60,40,20,10]
        # self.topk=[120,100,100,80,80,80,60,60,60]
        self.topk=[20,20,20,20]
        self.contextDim = [i ** 2 for i in self.topk]
        self.contextDim.insert(0, HyperParam.AutoIntFeatureNum)
        self.contextDim.pop(-1)
        self.depth = len(self.topk)
        self.headNum = HyperParam.AutoIntHeadNumList
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.transformer = nn.ModuleList([SparseDCNAttentionLayer(self.featureNum, self.contextDim[i],
                                                                  self.featureDim,
                                                                  self.topk[i])
                                          for i in range(self.depth)])
        self.output = None
        HyperParam.AutoIntMatchMlpDims[0] = reduce(lambda x, y: x + y ** 2, self.topk,HyperParam.AutoIntFeatureNum) * HyperParam.AutoIntFeatureDim
        # HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * sum(self.topk) * HyperParam.AutoIntFeatureDim
        self.mlp = AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)

    def mainForward(self, feature):
        query = feature
        context = feature
        result = [query]
        for i in range(self.depth):
            context = self.transformer[i](query, context)
            result.append(context)
        out = torch.cat(result, dim=1)
        result = self.mlp(None, None, None, None, None, None, out)
        return result
