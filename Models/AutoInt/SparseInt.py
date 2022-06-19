from collections import OrderedDict
from typing import List
from Models.Layers.SparseAttentionLayer import SparseAttentionLayer
from Models.BaseModelV2 import BaseModelV2, BaseEmbedFormat
from torch import nn
import torch
from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE
from Models.Modules.MatchModule import AllConcatMlpV2
from Models.Modules.FusionModule import MultiLayerTransformer


class SparseInt(BaseModelV2):
    def __init__(self, featureInfo: List[FeatureInfo], embedFormat: BaseEmbedFormat):
        self.nameList = [i.featureName for i in featureInfo if (
                i.enable == True and i.featureType == FEATURETYPE.USER)]
        super().__init__(embedFormat)
        self.featureNum = HyperParam.AutoIntFeatureNum
        # self.topk = [30, 20, 20, 10, 10]
        self.depth = len(self.topk)
        self.headNum = HyperParam.AutoIntHeadNumList
        self.featureDim = HyperParam.AutoIntFeatureDim
        self.transformer = nn.ModuleList([SparseAttentionLayer(self.featureNum, self.featureDim, self.topk[i])
                                          for i in range(self.depth)])
        self.output = None
        HyperParam.AutoIntMatchMlpDims[0] = HyperParam.AutoIntFeatureNum * (
                1 + sum(self.topk)) * HyperParam.AutoIntFeatureDim
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
