import os
import sys

from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format
import functools
from itertools import chain
from enum import Enum
import json
import os
from abc import abstractmethod
from typing import List
from datetime import datetime
import sys
# try:
#     from light import light, light_init
import torch
import typing
import random
import numpy as np
import re
from torch._six import container_abcs, string_classes, int_classes

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r'[SaUO]')

# sys.path.append("/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/ft_local/GMM_torch/")
# sys.path.append("/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/ft_local/GMM_torch/tfrecord/")
import torch
from Utils.HyperParamLoadModule import Config

if Config.fatherPid == 0:
    print("main.py pip :pid:" + os.getpid().__str__())
    os.system("pip install psutil")
    print("main.pypsutils:pid:" + os.getpid().__str__())
    os.system("pip install scikit-learn")

from Data.Input import TFDataset, default_collate

from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from datetime import datetime
from Models.GMM import *

print("main.py:pid:" + os.getpid().__str__())
from Models.AutoInt import *
from Data.Dataset import *
from Utils.Evaluation import Auc
from Utils.HyperParamLoadModule import *
from Utils.loss import *
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from datetime import datetime
from Models.GMM import *
import tensorflow as tf

print(tf.__version__)
print("main.py:pid:" + os.getpid().__str__())
from Models.AutoInt import *
from Data.Dataset import *
from Utils.Evaluation import Auc
from Utils.HyperParamLoadModule import *
from Utils.loss import *
from sklearn.metrics import roc_auc_score
import numpy as np
import psutil

import psutil
from Models.BaseModel import BaseModel
import multiprocessing
# from torchstat import stat
#
# sys.path.append('/home/baiting/anaconda3/lib/python3.7/site-packages')
from datetime import datetime
from Models.GMM import *
from DataSource.Avazu import *
from DataSource.Criteo import *
print("main.py:pid:" + os.getpid().__str__())
from Models.AutoInt import *
from Models.FiBiNet import *
from Models.XDeepFM import *
from Models.Lambda import *
from torch.utils.tensorboard import SummaryWriter
from Data.Dataset import *
from Utils.Evaluation import Auc
from Utils.HyperParamLoadModule import *
from Utils.loss import *
from sklearn.metrics import roc_auc_score
import numpy as np
from Models.BaseModel import BaseModel
import multiprocessing
# from torchstat import stat
#
# sys.path.append('/home/baiting/anaconda3/lib/python3.7/site-packages')
from datetime import datetime
from Models.GMM import *
from DataSource.WXBIZEmbed import *
from DataSource.WXBIZEmbed import *
from DataSource.BaseDataFormat import *
from DataSource import *
print("main.py:pid:" + os.getpid().__str__())
from Models.AutoInt import *
from torch.utils.tensorboard import SummaryWriter
from Data.Dataset import *
from Utils.Evaluation import Auc
from Utils.HyperParamLoadModule import *
from Utils.loss import *
from sklearn.metrics import roc_auc_score
import numpy as np
from Models.FiBiNet import *

from Models.BaseModelV2 import BaseModelV2


def load_model(featureInfo):
    preTrainModel = os.path.join(Config.absPath, Config.savedModelPath, Config.preTrainModelName)
    print(preTrainModel)
    if Config.buildState != BUILDTYPE.TRAIN and os.path.exists(preTrainModel) is False:
        raise Exception('running without model path!')
    # 加载模型（反射）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'GPU:  {torch.cuda.get_device_name(0)}')
        torch.cuda.set_device(0)

    embedFormat = eval(f"{Config.datasetType.name}{Config.embedType}.{Config.datasetType.name}{Config.embedType}")(
        BaseEmbedPack(HyperParam.popCatFeature, HyperParam.popSeqFeature, HyperParam.popDims2DimFeature, featureInfo))
    model: BaseModelV2 = eval(f'{Config.modelName}.{Config.modelName}')(featureInfo, embedFormat).to(device)

    loss_fn = eval(Config.lossType)().to(device)
    optimizer = torch.optim.Adam(model.buildParameterGroup(), lr=HyperParam.LR, weight_decay=HyperParam.L2)
    # 初始化参数/加载参数
    if (Config.loadPreTrainModel is True and os.path.exists(preTrainModel)):
        save_info = torch.load(preTrainModel)
        optimizer.load_state_dict(save_info['optimizer'])
        model.EmbedFormat.loadEmbedding()
    return model, loss_fn, optimizer, device


def model_train(model, loss_fn, optimizer, metrics: list, dataset: DatasetV2, epoch: int, device):
    path = os.path.join(Config.datasetPath, Config.logPath, Config.modelName, f"lr={HyperParam.LR},l2={HyperParam.L2}",
                        f'{datetime.now().timestamp().__str__()}')
    print(path)
    print(multiprocessing.cpu_count())
    name = ["vid_week_id", "vid_hour_id"]
    writer = SummaryWriter(path)
    print(f" chmod -R 777 {os.path.join(Config.datasetPath, Config.logPath)}")
    os.system(f" chmod -R 777 {os.path.join(Config.datasetPath, Config.logPath)}")
    # result=calFrequency(name, dataset.trainData, writer)
    aucBucket = Auc(102400)
    index = 0
    data, _ = next(iter(dataset.train.getBatchData()))
    for step in range(1000):
        print(f'epoch:{step}\n')
        # model.calBeforeEachEpoch(i)
        start = datetime.now()
        epochLoss = 0

        for feed_dict, k in dataset.train.getBatchData():
            end = datetime.now()
            device_dict = {}
            feed_dict = data
            for key, val in feed_dict.items():
                device_dict[key] = torch.as_tensor(val).to(device)

            print(f"reading time:batch:{k}, cost:{(end - start).total_seconds()} seconds\n")

            batchLoss = 0
            model.train()
            optimizer.zero_grad()
            prediction = model(device_dict)
            start1 = datetime.now()
            loss: torch.Tensor = loss_fn(prediction.squeeze(-1), device_dict['label'].squeeze(-1))
            lossAux = loss + model.getAuxLoss()
            lossAux.backward()
            optimizer.step()

            end1 = datetime.now()
            a = end1 - start1
            print(f"training time:batch:{index}, cost:{((end1 - start1).total_seconds())} seconds\n")

            if (index % 1 == 0):
                aucBucket.Reset()
                try:
                    start = datetime.now()
                    auc = roc_auc_score(device_dict['label'].cpu().numpy(), prediction.cpu().detach().numpy())
                    end = datetime.now()
                    print(f"auc time:batch:{index}, cost:{(end - start).total_seconds()} seconds\n")
                except ValueError:
                    auc = -1
                writer.add_scalar('loss/trainLoss', loss, global_step=index, walltime=None)
                print(f'batch {index}:  batchLoss:{loss}   auc: {auc}\n')
            start = datetime.now()
            # testing:
            # if (index % 500 == 0):
            #     print('valing......')
            #     start1 = datetime.now()
            #     aucVal, testLoss = model_test(model, dataset.train, loss_fn, True)
            #     end1 = datetime.now()
            #     print(f"val time:batch:{index}, cost:{((end1 - start1).total_seconds())} seconds\n")
            #     writer.add_scalar('auc/val', aucVal, global_step=index, walltime=None)
            #     writer.add_scalar('loss/valtest', testLoss, global_step=index, walltime=None)
            #     print(f"batch{k}:  aucVal:  {aucVal}")
            # if (index % 500 == 0):
            #     print('testing......')
            #     start1 = datetime.now()
            #     aucVal, testLoss = model_test(model, dataset.test, loss_fn, True)
            #     end1 = datetime.now()
            #     print(f"test time:batch:{index}, cost:{((end1 - start1).total_seconds())} seconds\n")
            #     writer.add_scalar('auc/Pintest', aucVal, global_step=index, walltime=None)
            #     writer.add_scalar('loss/Pintest', testLoss, global_step=index, walltime=None)
            #
            #     # save(os.path.join(Config.absPath, Config.savedModelPath) + f"{Config.modelName}.pt", optimizer, model)
            #
            #     print(f"batch{k}:  auctest:  {aucVal}")
            # if (index % 1000 == 0):
            #     print('testing......')
            #     start1 = datetime.now()
            #     aucVal, testLoss = model_test(model, dataset.test, loss_fn, False)
            #     end1 = datetime.now()
            #     print(f"test time:batch:{index}, cost:{((end1 - start1).total_seconds())} seconds\n")
            #     writer.add_scalar('auc/Alltest', aucVal, global_step=index, walltime=None)
            #     writer.add_scalar('loss/Alltest', testLoss, global_step=index, walltime=None)
            #
            #     save(os.path.join(Config.absPath,
            #                       Config.savedModelPath) + f"{Config.modelName}lr={HyperParam.LR},l2={HyperParam.L2}.pt",
            #          optimizer, model)
            #     print(f"batch{k}:  auctest:  {aucVal}")
            del device_dict
            index += 1
        aucVal, meanLoss = model_test(model, dataset.test, loss_fn, False)
        save(os.path.join(Config.absPath,
                          Config.savedModelPath) + f"{Config.modelName}lr={HyperParam.LR},l2={HyperParam.L2}.pt",
             optimizer, model)
        writer.add_scalar('auc/globalTest', aucVal, global_step=step, walltime=None)
        writer.add_scalar('loss/globalTest', meanLoss, global_step=step, walltime=None)
        writer.add_scalar('loss/epoch', epochLoss, global_step=step, walltime=None)


def model_test(model, dataset: BaseDataFormat, loss_fn, pinMemoryData=True):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        val = []
        count = 0
        truth = []
        testLoss = 0
        if pinMemoryData:
            data = dataset.getBufferData()
        else:
            data = dataset.getBatchData()
        start = datetime.now()
        for feed_dict, _count in data:
            count = _count
            new_feed = {}
            for i, v in feed_dict.items():
                new_feed[i] = torch.as_tensor(v).to(device)
            # end=datetime.now()
            # print(f"reading time:, cost:{(end - start).total_seconds()} seconds\n")
            # start=datetime.now()
            prediction = model(new_feed)
            loss: torch.Tensor = loss_fn(prediction.squeeze(-1), new_feed['label'].squeeze(-1))
            testLoss += loss.cpu().item()
            # end=datetime.now()
            # print(f"eval time:, cost:{(end - start).total_seconds()} seconds\n")
            val.append(prediction.cpu().numpy())
            truth.append(new_feed['label'].cpu().numpy())
            # start=datetime.now()
        # print(roc_auc_score(feed_dict['label'].cpu().numpy(), prediction.cpu().numpy()))
        try:
            # start=datetime.now()
            auc = roc_auc_score(np.concatenate(truth, axis=0).squeeze(), np.concatenate(val, axis=0).squeeze())
            meanLoss = testLoss / count
            end = datetime.now()
            print(f"auc time:, cost:{(end - start).total_seconds()} seconds\n")
        except ValueError:
            auc = -1
    return auc, meanLoss


def save(save_path, optimizer, model):
    assert (save_path is not None)
    save_info = {
        'optimizer': optimizer.state_dict(), 'model': model.state_dict()
    }
    torch.save(save_info, save_path)
    print(f'model saved in {save_path}')


# array = ['BPR', 'GCN', 'GRU4Rec', 'NCF', 'NNCF', 'PinSage', 'UserRNN']


def main_process(featureInfo):
    # choose Dataset from:
    # ml-100k, Grocery_and_Gourmet_Food, ml-1m

    model, loss_fn, optimizer, device = load_model(featureInfo)

    metrics = ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'CNDCG@10', 'CNDCG@5', 'UNDCG@10', 'UNDCG@5']

    best_HR5, best_scores, pre_HR5, rank_distribution, counter = 0, {}, 0, {}, {}
    dataset = DatasetV2(Config.datasetType.name, HyperParam.batchSize, prefetch=HyperParam.prefetch)

    if Config.buildState == BUILDTYPE.TRAIN:
        model_train(model, loss_fn, optimizer, metrics, dataset, 1, device=device)
    if Config.buildState == BUILDTYPE.TEST:
        model_test(model, dataset.trainData)
    if Config.needSave is True:
        save(os.path.join(Config.absPath, Config.savedModelPath, Config.preTrainModelName), optimizer, model)


if __name__ == '__main__':
    '''
    python3 -u ./main.py TopDownIADNN  
    cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/daily_train/Criteo/ Criteo  0.0001 0.00001 512 1000
    '''
    # modelName = TopDownIADNN
    # datasetName = sys.argv[3]
    # datasetPath = sys.argv[2]
    # lr = sys.argv[4]
    # l2 = sys.argv[5]
    # batchSize = sys.argv[6]
    # print(modelName, datasetName, datasetPath, lr, l2)
    torch.multiprocessing.set_start_method("spawn")
    # abs_address = f'/cephfs/group/wxplat-wxbiz-offline-datamining/evanxcwang/daily_train/{datasetName}/Config/FeatureConfig.json'
    now = datetime.now()
    abs_address = "./Config/FeatureConfig.json"
    featureInfo = loadArgs(abs_address)
    # Config.modelName = modelName
    # Config.datasetPath = datasetPath
    # Config.preTrainModelName = "LTEEluV2lr=0.0005,l2=1e-06.pt"
    print(Config)
    # Config.loadPreTrainModel = True
    # HyperParam.LR = float(lr)
    # HyperParam.L2 = float(l2)
    # HyperParam.batchSize = int(batchSize)
    HyperParam.AutoPruningFeatureL0 = 0.001

    main_process(featureInfo)
