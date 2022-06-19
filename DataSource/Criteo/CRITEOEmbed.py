import torch
import torch.nn as nn
from DataSource.BaseDataFormat import *
from Utils.HyperParamLoadModule import *
from DataSource.Avazu.AVAZUData import *
from itertools import chain
from enum import Enum
import logging
import os
import time
import sys
import dis
import codecs

# for tensorflow env
try:
    import tensorflow as tf

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S')
    logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger()
except:
    pass


class CRITEOEmbed(BaseEmbedFormat):
    def __init__(self, args: BaseEmbedPack):
        self.dataPath = Config.datasetPath

        self.popSeqFeature = args.popSeqFeature
        self.popDims2DimFeature = args.popDims2DimFeature
        self.popCatFeature = args.popCatFeature
        # droplist={}
        self.feature_names, self.feature_defaults, self.categorical_feature_counts, self.feature_types, self.feature_dim = self.build_feature_meta(
            os.path.join(self.dataPath, 'features.json'), os.path.join(self.dataPath, 'feature_index'))

    def lookup(self, featureName: str, content, embedding, device):
        temp = embedding[featureName](content.type(torch.LongTensor).to(device))
        return temp

    def preProcess(self, rawData: dict, embedding, ):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CatFeature = {key: DataPack(name=key, fieldType=FIELDTYPE.CAT,
                                    data=self.lookup(key, value,
                                                     embedding, device))
                      for key, value in rawData.items() if key != 'label'}
        return None, None, CatFeature

    def buildEmbedding(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding: nn.ModuleDict[str:nn.Embedding] = nn.ModuleDict()
        for feature, numb in self.categorical_feature_counts.items():
            embedding[feature] = nn.Embedding(numb + 1, HyperParam.AutoIntFeatureDim).to(
                device=device)
        # load pretrain Model
        if Config.loadPreTrainModel:
            pass
        # due with share embedding
        else:
            with torch.no_grad():
                for key, value in embedding.items():
                    nn.init.xavier_normal(value.weight, gain=1.414)

        return embedding

    def build_feature_meta(self, features_meta_file='features.json', features_dict_file='dict.txt'):
        ft_cat_counts = {}
        ft_types = {}
        ft_dims = {}
        ft_names = ['label']
        ft_defaults = [[0]]
        with tf.io.gfile.GFile(features_meta_file, 'r') as ftfile:
            feature_meta = json.load(ftfile)
            for ft_name, ft_type, ft_dim in feature_meta:
                ft_names.append(ft_name)
                ft_types[ft_name] = ft_type
                ft_dims[ft_name] = ft_dim
                if ft_type == 'CATEGORICAL':
                    ft_defaults.append([0])
                elif ft_type == 'NUMERIC':
                    ft_defaults.append([0.0])

        with tf.io.gfile.GFile(features_dict_file) as fm_file:
            logger.info('Reading features from file %s', features_dict_file)
            for feature in fm_file:
                ft = feature.strip().split('\1')
                feature_name = ft[0].strip()
                if ft_cat_counts.get(feature_name) is None:
                    ft_cat_counts[feature_name] = 1
                else:
                    ft_cat_counts[feature_name] += 1

        return ft_names, ft_defaults, ft_cat_counts, ft_types, ft_dims

    def loadEmbedding(self, savedPath, model):
        if os.path.exists(savedPath):
            save_info = torch.load(savedPath,map_location=torch.device('cpu'))
            assert 'model' in save_info.keys()
            model.load_state_dict(save_info['model'])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    abs_address = '/Users/ivringwang/Desktop/tencent/GMM_torch/Config/FeatureConfig.json'
    featureInfo = loadArgs(abs_address)
    path = "/DataSource/Avazu/all_data.csv"
    test = AVAZUData(path)
    dataIter = test.getBatchData()
    embed = AVAZUEmbed([], [], [], featureInfo=None)
    embedding = embed.buildEmbedding()
    for i, count in dataIter:
        for j in i.keys():
            i[j] = torch.as_tensor(i[j]).to(device)
        result = embed.preProcess(i, embedding)
        print(result)
