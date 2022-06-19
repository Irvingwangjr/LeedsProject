# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.
from DataSource.BaseDataFormat import *
import tensorflow as tf
import logging
import json
import os
import numpy as np
from enum import Enum
import csv
import torch

logger = logging.getLogger(__name__)


class DATATYPE(Enum):
    test = 1
    train = 2


class AVAZUData(BaseDataFormat):
    def __init__(self, data_path,
                 batch_size=2048,
                 num_worker=22,
                 buffer_size=10000,
                 prefetch=100):
        super(AVAZUData, self).__init__()
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.data_path = data_path
        self.getFileList()

        self.feature_names = ['label', 'weekday', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
                              'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model',
                              'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        self.feature_defaults = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
                                 [0], [0], [0], [0], [0], [0], [0]]

        self.getPrefetch()

    def getFileList(self):
        self.fileList = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path)]
        if len(self.fileList) > 1:
            self.fileType = DATATYPE.test
            if 'test_data.csv' in self.fileList[0]:
                self.valFile = self.fileList[1]
                self.testFile = self.fileList[0]
            else:
                self.valFile = self.fileList[0]
                self.testFile = self.fileList[1]
        else:
            self.fileType = DATATYPE.train
            self.trainFile = self.fileList[0]

    def getPrefetch(self):
        if self.fileType == DATATYPE.test:
            self.valData = []
            for data, count in self.readFromFile(self.valFile):
                self.valData.append((data, count))
                break
                print(f"val {count}")
            print(f"valFinish{len(self.valData)}")

            self.testData = []
            for data, count in self.readFromFile(self.testFile):
                self.testData.append((data, count))
                print(f"test {count}")
                break
            print(f"testFinish{len(self.testData)}")

        if self.fileType == DATATYPE.train:
            self.trainBufferData = []
            self.trainData = []
            for data, count in self.readFromFile(self.trainFile):
                self.trainBufferData.append((data, count))
                if count >= self.buffer_size:
                    break
            for data, count in self.readFromFile(self.trainFile):
                print(f"train {count}")

                self.trainData.append((data, count))
                break

            print(f"trainFinish{len(self.trainData)}")

    def loadIntoMem(self, fileName):
        with open(fileName, 'r') as file:
            lines = csv.reader(file)
            out = list(lines)
        return out

    def readFromFile(self, fileName):
        dataset = tf.data.TextLineDataset(fileName)
        dataset = dataset.map(lambda x: self.parse_record(x, self.feature_names, self.feature_defaults),
                              num_parallel_calls=self.num_worker)
        dataset = dataset.batch(self.batch_size).prefetch(self.prefetch)
        count = 0
        for data in dataset.as_numpy_iterator():
            try:
                res_lt = data
                count += res_lt[1].shape[0]
                res_lt[0]['label'] = res_lt[1]
                res_lt[0]['label'] = res_lt[0]['label'].astype(np.float32)
                yield res_lt[0], count
            except GeneratorExit:
                print("generator exit")
                break
        del dataset

    def getBatchData(self):
        if self.fileType == DATATYPE.train:
            dataIter = self.trainData
        else:
            dataIter = self.testData
        count = 0
        for data in dataIter:
            try:
                res_lt = data
                count += res_lt[1]
                yield res_lt[0], count
            except GeneratorExit:
                print("generator exit")
                break

    def getBufferData(self):
        if self.fileType == DATATYPE.test:
            dataIter = self.valData
        else:
            dataIter = self.trainData
        count = 0
        for data in dataIter:
            try:
                res_lt = data
                count += res_lt[1]
                yield res_lt[0], count
                if self.fileType == DATATYPE.train and count >= self.buffer_size:
                    print("buffer finish")
                    break
            except GeneratorExit:
                print("generator exit")
                break

    def load_cross_fields(self, cross_fields_file):
        if cross_fields_file is None:
            return None
        else:
            return set(json.load(open(cross_fields_file)))

    # Create a feature
    def parse_record(self, record, feature_names, feature_defaults):
        feature_array = tf.io.decode_csv(record, feature_defaults)
        features = dict(zip(feature_names, feature_array))
        label = features.pop('label')
        # features.pop('tag') # unused
        # if features['ads_category'] < 0:
        #     features['ads_category'] = 0
        return features, label


if __name__ == "__main__":
    test = AVAZUData("all_data.csv", 64, 0, 10000, 0)
    dataIter = test.getBufferData()
    for i, count in dataIter:
        print(count)
