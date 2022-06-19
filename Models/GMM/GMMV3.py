from Models.BaseModel import BaseModel
from Models.GMM.GMM import GMM
from Models.Modules.FusionModule import *
from Models.Modules.MatchModule import *
from Models.Modules.CatTransform import LinearContextTrans
from Models.Modules.DIms2DimTransModule import LinearDimsTransModule
from Models.Modules.SeqTransform import GRULinearDimsTransModule

from Utils.HyperParamLoadModule import HyperParam, FeatureInfo


class GMMV3(GMM):
    def __init__(self, featureInfo: List[FeatureInfo], device):
        super(GMMV3, self).__init__(featureInfo, device)

    def loadFusionModule(self):
        return MultiLayerGaussianFusionModule(self.hyperParams.GMMV3GaussianFusionFeatureNumb,
                                              self.hyperParams.GMMV3GaussianFusionLayersDims,
                                              self.hyperParams.GMMV3GaussianFusionMlpLayerDims,
                                              self.hyperParams.GMMV3GaussianFusionLayerNumb)

    ## sideInfo:dense/:处理纬度
    def loadContextTransModule(self):
        return LinearContextTrans([HyperParam.FeatValDenseFeatureDim], HyperParam.AutoIntFeatureDim)

    ## None纬度变换
    def loadItemTransModule(self):
        return LinearDimsTransModule(HyperParam.ItemFeatValVecFeatureDims, HyperParam.AutoIntFeatureDim, )

    # None：纬度变换
    def loadUserTransModule(self):
        return GRULinearDimsTransModule(HyperParam.UserFeatValVecFeatureDims,
                                        HyperParam.AutoIntFeatureDim,
                                        HyperParam.AutoIntFeatureDim * len(self.nameList), self.nameList,
                                        HyperParam.SequenceLenKey)

    ##concatMLP
    def loadMatchingModule(self):
        return AllConcatMlpV2(HyperParam.AutoIntMatchMlpDims)
