from typing import List, Any

from Models.BaseModel import BaseModel
from Models.GMM.GMM import GMM
from Models.Modules.FusionModule import *
from Models.Modules.MatchModule import *

from Utils.HyperParamLoadModule import FeatureInfo, HyperParam, FEATURETYPE


class GMMV2(GMM):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        super(GMMV2, self).__init__(featureInfo, device)

    # def getParameter(self):
    #     a = [{"params": i} for i in self.embedding.values()]
    #     a.append({"params":self.userTransModule.})

    def loadMatchingModule(self):
        return AllConcatMlp(self.hyperParams.GMMV2AllMLPLayerDims, None, self.concatMatchHooker)

    def concatMatchHooker(self, userFeature: dict, userTrans, itemFeature: dict, itemTrans, contextFeature: dict,
                          contextTrans, fusionFeature):

        # 丢掉序列：
        userFeatureTemp = {}
        for i in userFeature.keys():
            if ("uin_vid" not in i and "label" not in i):
                userFeatureTemp[i] = userFeature[i]
        # print(
        #     f"userFeature: {len(userFeatureTemp)}, itemFeature: {len(itemFeature)},contextFeature: {len(contextFeature)}")
        result = dict(list(userFeatureTemp.items()) + list(itemFeature.items()) + list(contextFeature.items()) + list(
            {'userTrans': userTrans, 'itemTrans': itemTrans}.items()))
        for (i, tensor) in enumerate(fusionFeature):
            if tensor.ndim>2:
                result[f'fusion{i}'] = tensor.squeeze()
        return result
