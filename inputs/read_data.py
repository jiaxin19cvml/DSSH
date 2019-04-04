import os.path as op
import numpy as np
import h5py

def loadh5(gb, MODALITY, USAGE=None):
    if gb.DATASET=='SHREC13' or gb.DATASET=='SHREC14':
        if USAGE==None:
            h5filename = gb.DATASET + "_" + MODALITY + "_pool5_inception_viewCoded.hdf5"
            fin = h5py.File(op.join(gb.ROOT_DIR, "data", gb.DATASET, h5filename), 'r')
            feat = np.asarray(fin["feat_" + MODALITY])
            labels = np.asarray(fin["labels_" + MODALITY])
            sampleids = np.asarray(fin["sampleids_" + MODALITY])
            fids = list(fin["fids_" + MODALITY])
        else:
            h5filename = gb.DATASET + "_" + MODALITY + "_" + USAGE + "_pool5_inception.hdf5"
            fin = h5py.File(op.join(gb.ROOT_DIR, "data", gb.DATASET, h5filename), 'r')
            feat = np.asarray(fin["feat_" + MODALITY + "_" + USAGE])
            labels = np.asarray(fin["labels_" + MODALITY + "_" + USAGE])
            sampleids = np.asarray(fin["sampleids_" + MODALITY + "_" + USAGE])
            fids = list(fin["fids_" + MODALITY + "_" + USAGE])
        if MODALITY == "shape":
            projViews = np.asarray(fin["projViews_coded"])
            projViews = np.asarray( list(zip(*projViews)) )
            projViews = projViews.transpose([1,0,2])

            return feat, labels, sampleids, fids, projViews
        elif MODALITY == "sketch":
            return feat, labels, sampleids, fids
    elif gb.DATASET=='PART-SHREC14':
        if USAGE==None:
            h5filename = "PART_"+gb.DATASET + "_" + MODALITY + "_pool5_inception_viewCoded.hdf5"
            fin = h5py.File(op.join(gb.ROOT_DIR, "data", gb.DATASET, h5filename), 'r')
            feat = np.asarray(fin["feat_" + MODALITY])
            labels = np.asarray(fin["labels_" + MODALITY])
            sampleids = np.asarray(fin["sampleids_" + MODALITY])
            fids = list(fin["fids_" + MODALITY])
        else:
            h5filename = "PART_"+gb.DATASET + "_" + MODALITY + "_" + USAGE + "_pool5_inception.hdf5"
            fin = h5py.File(op.join(gb.ROOT_DIR, "data", gb.DATASET, h5filename), 'r')
            feat = np.asarray(fin["feat_" + MODALITY + "_" + USAGE])
            labels = np.asarray(fin["labels_" + MODALITY + "_" + USAGE])
            sampleids = np.asarray(fin["sampleids_" + MODALITY + "_" + USAGE])
            fids = list(fin["fids_" + MODALITY + "_" + USAGE])
        if MODALITY == "shape":
            projViews = np.asarray(fin["projViews_coded"])
            projViews = np.asarray( list(zip(*projViews)) )
            projViews = projViews.transpose([1,0,2])

            return feat, labels, sampleids, fids, projViews
        elif MODALITY == "sketch":
            return feat, labels, sampleids, fids
