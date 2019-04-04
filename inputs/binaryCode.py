import scipy
from scipy import ndimage
from scipy import misc
import random
try:
    import queue
except ImportError:
    import Queue as queue
import numpy as np
import os.path as op
import math
import time
import tensorflow as tf

"""***********************************************************************************************************
sample k fids for classid with siamese
***********************************************************************************************************"""


def sample_K_data_for_classid_siamese(gb, sampled_labels, feat_shape, labels_shape, sampleids_shape, projViews,\
                    feat_sketch, labels_sketch, sampleids_sketch):
    b_feat_shape = np.zeros( (gb.BATCH_K * gb.BATCH_P, gb.NUMB_CHANNELS, feat_shape.shape[2]))
    b_labels_shape = np.zeros(gb.BATCH_K * gb.BATCH_P)
    b_sampleids_shape = np.zeros(gb.BATCH_K * gb.BATCH_P)
    b_projViews = np.zeros((gb.BATCH_K * gb.BATCH_P, gb.NUMB_CHANNELS, 300))

    b_feat_sketch = np.zeros((gb.BATCH_K * gb.BATCH_P, feat_sketch.shape[1]))
    b_labels_sketch = np.zeros(gb.BATCH_K * gb.BATCH_P)
    b_sampleids_sketch = np.zeros(gb.BATCH_K * gb.BATCH_P)

    for i in range(len(sampled_labels)):
        possible_shape = np.where(labels_shape == sampled_labels[i])
        possible_sketch = np.where(labels_sketch == sampled_labels[i])
        possible_shape = possible_shape[0]
        possible_sketch = possible_sketch[0]

        count_shape = len(possible_shape)
        count_sketch = len(possible_sketch)
        count_padded_shape = math.ceil(gb.BATCH_K / count_shape) * count_shape
        count_padded_sketch = math.ceil(gb.BATCH_K / count_sketch) * count_sketch
        range_full_shape = np.arange(count_padded_shape)
        range_full_sketch = np.arange(count_padded_sketch)
        np.random.shuffle(range_full_shape)
        np.random.shuffle(range_full_sketch)
        range_full_shape = np.mod(range_full_shape, count_shape)
        range_full_sketch = np.mod(range_full_sketch, count_sketch)

        selected_indx_shape = possible_shape[range_full_shape[:gb.BATCH_K]]
        b_feat_shape[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = feat_shape[selected_indx_shape]
        b_labels_shape[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = labels_shape[selected_indx_shape]
        b_sampleids_shape[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = sampleids_shape[selected_indx_shape]
        b_projViews[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = projViews[selected_indx_shape]

        selected_indx_sketch = possible_sketch[range_full_sketch[:gb.BATCH_K]]
        b_feat_sketch[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = feat_sketch[selected_indx_sketch]
        b_labels_sketch[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = labels_sketch[selected_indx_sketch]
        b_sampleids_sketch[i*gb.BATCH_K:(i+1)*gb.BATCH_K] = sampleids_sketch[selected_indx_sketch]

    return b_feat_shape, b_labels_shape, b_sampleids_shape, b_projViews,\
            b_feat_sketch, b_labels_sketch, b_sampleids_sketch


"""
buildDataset
"""
class buildTrainData:
    def __init__(self, gb, feat_shape, labels_shape, sampleids_shape, projViews,\
                    feat_sketch_train, labels_sketch_train, sampleids_sketch_train):
        self.gb = gb
        self.feat_shape = feat_shape
        self.labels_shape = labels_shape
        self.sampleids_shape = sampleids_shape
        self.projViews = projViews
        self.feat_sketch_train = feat_sketch_train
        self.labels_sketch_train = labels_sketch_train
        self.sampleids_sketch_train = sampleids_sketch_train



    def batches(self):
        unique_labels = np.unique(self.labels_shape)
        unique_labels.astype('int32')
        numb_sampledClass = math.floor(len(unique_labels) / self.gb.BATCH_P) * self.gb.BATCH_P
        np.random.shuffle(unique_labels)
        sampled_unique_labels = unique_labels[:numb_sampledClass]
        for i in range(0, numb_sampledClass, self.gb.BATCH_P):
            sampled_labels = sampled_unique_labels[i:i + self.gb.BATCH_P]
            b_feat_shape, b_labels_shape, b_sampleids_shape, b_projViews,\
            b_feat_sketch, b_labels_sketch, b_sampleids_sketch = \
                sample_K_data_for_classid_siamese(self.gb, sampled_labels, self.feat_shape,
                                                  self.labels_shape, self.sampleids_shape, self.projViews,
                                                  self.feat_sketch_train, self.labels_sketch_train, self.sampleids_sketch_train)

            yield b_feat_shape, b_labels_shape, b_sampleids_shape, b_projViews,\
                    b_feat_sketch, b_labels_sketch, b_sampleids_sketch



class BinaryCodes:
    def __init__(self, gb, shapeids, sketchids, labels_shape, labels_sketch):
        self.gb = gb
        unique_shapeids = np.unique(shapeids)
        unique_labels_shape = np.zeros(len(unique_shapeids))
        for i in range(len(unique_shapeids)):
            temp_indx = np.where(shapeids==unique_shapeids[i])
            temp_indx=temp_indx[0]
            unique_labels_shape[i] = labels_shape[temp_indx[0]]

        shapeidIdxMap = dict(zip(unique_shapeids,np.arange(0,len(unique_shapeids)) ))
        sketchid2BIndxMap = dict(zip(sketchids, np.arange(0, len(sketchids))))
        B_shape = np.zeros((len(unique_shapeids), self.gb.EMD_DIM), dtype="float32")
        B_sketch = np.zeros((len(labels_sketch), self.gb.EMD_DIM), dtype="float32")
        self.B_shape = B_shape
        self.B_sketch = B_sketch
        self.shapeid2BIndxMap = shapeidIdxMap
        self.sketchid2BIndxMap = sketchid2BIndxMap
        self.labels_of_unique_shapes = unique_labels_shape
        self.labels_of_unique_sketch = labels_sketch
    def init_B_perfect( self ):
        unique_labels_shape = np.unique( self.labels_of_unique_shapes )
        shapeLabels2BIndxMap_shape = dict( zip(unique_labels_shape, np.arange(0, len(unique_labels_shape))) )
        if False:#len(unique_labels_shape)<=self.gb.EMD_DIM:
            eyeMat = np.eye( self.gb.EMD_DIM, dtype="float32")
            eyeMat[np.where(eyeMat<0.5)]=-1.0
            B_essence = eyeMat[0:len(unique_labels_shape),:]
        else:
            for i in range(len(unique_labels_shape)):
                if i==0:
                    B_essence = np.array([(np.random.randint(0,2, (self.gb.EMD_DIM))-0.5)*2])
                else:
                    flag=False
                    counter = 0
                    while(~flag):
                        flag = True
                        b = np.array([(np.random.randint(0,2, (self.gb.EMD_DIM))-0.5)*2])
                        for j in range(B_essence.shape[0]):
                            if np.sum(np.abs(b-B_essence[j]))<0.1:
                                flag=False
                                counter+=1
                                break
                        if flag:
                            B_essence = np.concatenate((B_essence, b), axis=0)
                            break
                        if counter>20000:
                            raise ValueError("Being stacked when generating binary codes")

        indx_B_shape = np.array( [shapeLabels2BIndxMap_shape[i] for i in list(self.labels_of_unique_shapes)] )#np.array( map(lambda i: labelsIndxMap[i], self.labels_unique_shape) )
        indx_B_shape.astype("int32")
        self.B_shape = B_essence[indx_B_shape,:]
        indx_B_sketch = np.array( [shapeLabels2BIndxMap_shape[i] for i in list(self.labels_of_unique_sketch)] )
        indx_B_sketch.astype("int32")
        self.B_sketch = B_essence[indx_B_sketch, :]
        self.B_essence = B_essence
        self.shapeLabels2BIndxMap_shape = shapeLabels2BIndxMap_shape
















