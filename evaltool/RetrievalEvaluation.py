import tensorflow as tf
import numpy as np

from models import DSSH as model


def sigmod(x):
    x = np.array(x)
    x = 1 / (1+np.exp(-x))
    return x

def softmax(x):
    x = np.array(x)
    x = np.exp(x)
    sumcol =x.sum(axis=1)
    summat = np.expand_dims(sumcol, 1)
    summat = np.repeat(summat, x.shape[1], 1)
    x = x / summat
    return x


def tanh(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y


def computeMatEucDist( matA, matB ):
    matA[np.where( matA>=0 )] = 1
    matA[np.where( matA<0)] = -1
    matB[np.where(matB >= 0)] = 1
    matB[np.where(matB < 0)] = -1

    squaA = np.tile( np.matrix(np.sum(matA**2,axis=1)).transpose(), (1, matB.shape[0]) )
    squaB = np.tile(np.sum(matB**2, axis=1).transpose(), (matA.shape[0], 1))
    distM = np.sqrt(squaA - 2 * np.dot(matA, matB.transpose()) + squaB)

    return distM

def computerACC( matA, labelA ):
    matA = softmax(matA)
    correct_prediction = np.equal(np.argmax(matA, 1), labelA)
    correct_prediction.astype("float32")
    acc = np.mean(correct_prediction)
    return acc


def RetrievalEvaluation(feat_model, model_label, fids_model, feat_depth, depth_label, fids_depth, testMode=1):

    C_depth = np.zeros((len(depth_label)), dtype="int32")
    for i in range(len(depth_label)):
        C_depth[i] = np.sum(model_label == depth_label[i])

    distM = computeMatEucDist(feat_depth, feat_model)

    if testMode == 1:
        C = C_depth
        recall = np.zeros((distM.shape[0], distM.shape[1]))
        precision = np.zeros((distM.shape[0], distM.shape[1]))
        rankArray = np.zeros((distM.shape[0], distM.shape[1]))
    elif testMode == 2:
        C = C_depth - 1
        recall = np.zeros((distM.shape[0], distM.shape[1]-1))
        precision = np.zeros((distM.shape[0], distM.shape[1]-1))
        rankArray = np.zeros((distM.shape[0], distM.shape[1]-1))

    nb_of_query = C.shape[0]
    p_points = np.zeros((nb_of_query, np.amax(C)))
    ap = np.zeros(nb_of_query)
    nn = np.zeros(nb_of_query)
    ft = np.zeros(nb_of_query)
    st = np.zeros(nb_of_query)
    dcg = np.zeros(nb_of_query)
    e_measure = np.zeros(nb_of_query)


    for qqq in range(nb_of_query):
        temp_dist = np.array(distM[qqq])
        temp_dist = np.squeeze(temp_dist)
        s = list(temp_dist)
        R = sorted(range(len(s)), key=lambda k: s[k])
        if testMode == 1:
            model_label_l = model_label[R]
            numRetrieval = distM.shape[1]
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R
        elif testMode == 2:
            model_label_l = model_label[R[1:]]
            numRetrieval = distM.shape[1] - 1
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R[1:]

        model_label_l = np.squeeze(model_label_l)
        for i in range(numRetrieval):
            if model_label_l[i] == depth_label[qqq]:
                G[i] = 1
        G_sum = np.cumsum(G)
        r1 = G_sum / float(C[qqq])
        p1 = G_sum / np.arange(1, numRetrieval+1)
        r_points = np.zeros(C[qqq])
        for i in range(C[qqq]):
            temp = np.where(G_sum == i+1)
            r_points[i] = np.where(G_sum == (i+1))[0][0] + 1
        r_points_int = np.array(r_points, dtype=int)

        p_points[qqq][:int(C[qqq])] = G_sum[r_points_int-1] / r_points
        ap[qqq] = np.mean(p_points[qqq][:int(C[qqq])])
        nn[qqq] = G[0]
        ft[qqq] = G_sum[C[qqq]-1] / C[qqq]
        st[qqq] = G_sum[min(2*C[qqq]-1, G_sum.size-1)] / C[qqq]
        p_32 = G_sum[min(31, G_sum.size-1)] / min(32, G_sum.size)
        r_32 = G_sum[min(31, G_sum.size-1)] / C[qqq]
        if p_32 == 0 and r_32 == 0:
            e_measure[qqq] = 0
        else:
            e_measure[qqq] = 2* p_32 * r_32/(p_32+r_32)

        if testMode == 1:
            NORM_VALUE = 1 + np.sum(1/np.log2(np.arange(2,C[qqq]+1)))
            dcg_i = 1/np.log2(np.arange(2, len(R)+1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0)/NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1
        elif testMode == 2:
            NORM_VALUE = 1 + np.sum(1/np.log2(np.arange(2,C[qqq]+1)))
            dcg_i = 1/np.log2(np.arange(2, len(R[1:])+1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0)/NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1

    nn_av = np.mean(nn)
    ft_av = np.mean(ft)
    st_av = np.mean(st)
    dcg_av = np.mean(dcg)
    e_av = np.mean(e_measure)
    map_ = np.mean(ap)

    pre = np.mean(precision, axis=0)
    rec = np.mean(recall, axis=0)

    return nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray


def extractFeat(gb, sess, feat, labels, sampleids, fids, MODALITY, sampletime=3, projViews=None):
    # load image and trained model
    print("\n\t extract features...")
    if MODALITY=="SKETCH":
        b_test = tf.placeholder('float32', shape=(None, feat.shape[1]) )
        net_test = model.sketchHashNet(b_test, gb, is_training=False, is_reuse=True)
        emb = np.zeros((len(labels), gb.EMD_DIM), np.float32)
        for start_idx in range(0, len(labels), gb.BATCH_SIZE_TEST_SKETCH):
            end_idx = np.min( (start_idx+gb.BATCH_SIZE_TEST_SKETCH, len(labels)) )
            feed_dict = {b_test: feat[start_idx:end_idx]}
            b_emb = sess.run( net_test['emb'], feed_dict=feed_dict )
            emb[start_idx:start_idx + len(b_emb)] = b_emb
            print('\rEmbedded batch {}-{}/{}\tSampletime: {}'.format( start_idx, start_idx + len(b_emb), len(emb), sampletime+1), flush=True, end='')
        uniqueSampleids = np.unique(sampleids)
        if (len(uniqueSampleids) < len(sampleids)):
            labels_temp = np.array([])
            fids_temp = []
            for i in range(len(uniqueSampleids)):
                indx = np.array(np.where(sampleids == uniqueSampleids[i]))
                indx = np.squeeze(indx)
                indx.astype("int32")
                if i == 0:
                    emb_temp = np.array([np.mean(emb[indx], axis=0)])
                    labels_temp = np.array([labels[indx[0]]])
                    fids_temp.append(fids[indx[0]])
                else:
                    emb_temp = np.concatenate((emb_temp, np.array([np.mean(emb[indx], axis=0)])))
                    labels_temp = np.concatenate((labels_temp, np.array([labels[indx[0]]])))
                    fids_temp.append(fids[indx[0]])
            emb = emb_temp
            labels = np.squeeze(labels_temp)
            fids = fids_temp

            return emb, labels, fids

    elif MODALITY=="SHAPE":
        b_test = tf.placeholder('float32', shape=(None, gb.NUMB_CHANNELS, feat.shape[2]))
        b_projViews = tf.placeholder('float32', shape=(None, gb.NUMB_CHANNELS, 300))
        net_test = model.shapeHashNet(b_test, b_projViews, gb, is_training=False, is_reuse=True )
        emb = np.zeros((len(labels), gb.EMD_DIM), np.float32)
        for start_idx in range(0, len(labels), gb.BATCH_SIZE_TEST_SHAPE):
            end_idx = np.min( (start_idx+gb.BATCH_SIZE_TEST_SHAPE, len(labels)) )
            feed_dict = {b_test: feat[start_idx:end_idx],
                         b_projViews: projViews[start_idx:end_idx]}
            b_emb = sess.run( net_test['emb'], feed_dict=feed_dict )
            emb[start_idx:start_idx + len(b_emb)] = b_emb
            print('\rEmbedded batch {}-{}/{}\tSampletime: {}'.format( start_idx, start_idx + len(b_emb), len(emb), sampletime+1), flush=True, end='')

        uniqueSampleids = np.unique( sampleids )
        if( len(uniqueSampleids)<len(sampleids) ):
            labels_temp = np.array([])
            fids_temp = []
            for i in range(len(uniqueSampleids)):
                indx = np.array(np.where(sampleids == uniqueSampleids[i]))
                indx = np.squeeze(indx)
                indx = np.random.permutation(indx)
                indx.astype("int32")
                if i==0:
                    emb_temp = np.array([np.mean(emb[indx[0:sampletime]], axis=0)])
                    labels_temp = np.array([labels[indx[0]]])
                    fids_temp.append(fids[indx[0]])
                else:
                    emb_temp = np.concatenate( (emb_temp, np.array([np.mean(emb[indx[0:sampletime]], axis=0)])) )
                    labels_temp = np.concatenate( (labels_temp, np.array([labels[indx[0]]])))
                    fids_temp.append(fids[indx[0]])
            emb = emb_temp
            labels = np.squeeze(labels_temp)
            fids = fids_temp

            return emb, labels, fids
    else:
        raise ValueError("Input modality should be either SHAPE or SKETCH.")


def computeRetrievalMetrics(gb, sess, feat_shape, labels_shape, sampleids_shape, projViews, fids_shape,\
                        feat_sketch_train, labels_sketch_train, sampleids_sketch_train, fids_sketch_train,\
                        feat_sketch_test, labels_sketch_test, sampleids_sketch_test, fids_sketch_test, sampletime=3):

    feat_shape, labels_shape, fids_shape = \
        extractFeat(gb, sess, feat_shape, labels_shape, sampleids_shape, fids_shape, "SHAPE", sampletime,  projViews)
    feat_train_sketch, labels_train_sketch, fids_sketch_train = \
        extractFeat(gb, sess, feat_sketch_train, labels_sketch_train, sampleids_sketch_train, fids_sketch_train, "SKETCH", sampletime )
    feat_test_sketch, labels_test_sketch, fids_sketch_test = \
        extractFeat(gb, sess, feat_sketch_test, labels_sketch_test, sampleids_sketch_test, fids_sketch_test, "SKETCH", sampletime)

    nn_test, ft_test, st_test, dcg_test, e_test, map_test, p_points_test, pre_test, rec_test, rankArray_test = RetrievalEvaluation(feat_shape, labels_shape, fids_shape, feat_test_sketch, labels_test_sketch, fids_sketch_test)
    nn_train, ft_train, st_train, dcg_train, e_train, map_train, p_points_train, pre_train, rec_train, rankArray_train = RetrievalEvaluation(feat_shape, labels_shape, fids_shape, feat_train_sketch, labels_train_sketch, fids_sketch_train)

    # retrieve list
    rankArray_test = rankArray_test.astype(int)
    rankList_test = []
    for i in range(rankArray_test.shape[0]):
        rankList_test.append([fids_shape[p] for p in rankArray_test[i,...]])

    acc_test_sketch = computerACC( feat_test_sketch, labels_test_sketch )
    acc_train_sketch = computerACC(feat_train_sketch, labels_train_sketch)
    acc_shape = computerACC(feat_shape, labels_shape)

    return nn_test, ft_test, st_test, dcg_test, e_test, map_test, p_points_test, pre_test, rec_test,\
           nn_train, ft_train, st_train, dcg_train, e_train, map_train, p_points_train, pre_train, rec_train,\
            acc_test_sketch, acc_train_sketch, acc_shape, fids_sketch_test, rankList_test

