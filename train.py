import os
import os.path as op
from datetime import timedelta
from signal import SIGINT, SIGTERM
import time
import numpy as np

import tensorflow as tf

from utils import lbtoolbox as lb
from evaltool.RetrievalEvaluation import computeRetrievalMetrics

from inputs.read_data import loadh5
from inputs.binaryCode import buildTrainData, BinaryCodes
from models import DSSH as model
from configs import SHREC13 as gb
slim = tf.contrib.slim


"""
define the main() function
"""
def main():
    #****************
    #*  load data
    #*****************
    print("Load data...")
    feat_shape, labels_shape, sampleids_shape, fids_shape, projViews = loadh5(gb, "shape")
    feat_sketch_test, labels_sketch_test, sampleids_sketch_test, fids_sketch_test = loadh5(gb, "sketch", USAGE="test")
    feat_sketch_train, labels_sketch_train, sampleids_sketch_train, fids_sketch_train = loadh5(gb, "sketch", USAGE="train")

    db_train = buildTrainData( gb, feat_shape, labels_shape, sampleids_shape, projViews, \
                        feat_sketch_train, labels_sketch_train, sampleids_sketch_train)
    #**********************
    #*  build binary codes
    #**********************
    B = BinaryCodes(gb, sampleids_shape, sampleids_sketch_train, labels_shape, labels_sketch_train)
    B.init_B_perfect()

    with tf.Graph().as_default():
        # placeholders for graph input
        b_feat_shape = tf.placeholder('float32', shape=(None, gb.NUMB_CHANNELS, feat_shape.shape[2]), name='shape')
        b_feat_sketch = tf.placeholder('float32', shape=(None, feat_sketch_train.shape[1]), name='sketch')
        b_B_shape = tf.placeholder('float32', shape=(None, gb.EMD_DIM), name='B_shape')
        b_B_sketch = tf.placeholder('float32', shape=(None, gb.EMD_DIM), name='B_sketch')
        b_label_shape = tf.placeholder('int32', shape=[None], name='label_shape')
        b_label_sketch = tf.placeholder('int32', shape=[None], name='label_sketch')
        b_sampleids_shape = tf.placeholder('int32', shape=[None], name='sampleids_shape')
        b_sampleids_sketch = tf.placeholder('int32', shape=[None], name='sampleids_sketch')
        b_projViws = tf.placeholder('float32', shape=(None, gb.NUMB_CHANNELS, 300), name='projViews')
        mask = tf.placeholder('float32', shape=(None, None), name='mask')
        #********************************
        #*  define the model structure
        #*********************************
        print("Build model...")
        net_shape = model.shapeHashNet( b_feat_shape, b_projViws, gb, is_training=True, is_reuse=False)
        net_sketch = model.sketchHashNet( b_feat_sketch, gb, is_training=True, is_reuse=False)
        b_feat_shape_pooled = net_shape["input"]

        val_loss, loss_shape, loss_sketch = \
            model.loss_SVP_siamese( net_shape['emb_vp'], net_sketch['emb'], b_label_shape,  b_label_sketch, b_B_shape, b_B_sketch, b_feat_shape_pooled, b_feat_sketch, mask, gb )

        #********************************
        #*  define the optimizer and variables to be trained
        #*********************************
        print("Build optimizer...")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_var = tf.trainable_variables()
        if 0 <= gb.DECAY_STAT_ITERATION < gb.MAX_NUM_ITER:
            learning_rate = tf.train.exponential_decay(
                gb.LEARNING_RATE,
                tf.maximum(0, global_step - gb.DECAY_STAT_ITERATION),
                gb.MAX_NUM_ITER - gb.DECAY_STAT_ITERATION, 0.001)
        else:
            learning_rate = gb.LEARNING_RATE
        optimizer = tf.train.AdamOptimizer(learning_rate)
        var_finetune = train_var
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(val_loss, var_list=var_finetune, global_step=global_step)

        #********************************
        #*  define the initializer
        #********************************
        print("Build initializer...")
        init_op = tf.global_variables_initializer()
        checkpoint_saver = tf.train.Saver(max_to_keep=0)

        #*******************
        # define the logger
        #*******************
        if not op.isdir(op.join(gb.EXP_ROOT)):
            try:
                os.makedirs(op.join(gb.EXP_ROOT))
            except:
                pass
        log_file = open(op.join(gb.EXP_ROOT, "train.txt"),'w')
        log_file.write('Training using the following parameters:\n')
        for key, value in sorted(vars(gb).items()):
            log_file.write('{}: {}\n'.format(key, value))
        log_file.write( 'B_essence:\n' )
        for i in range( B.B_essence.shape[0] ):
            for j in range( B.B_essence.shape[1] ):
                log_file.write( "%d "%(B.B_essence[i][j]) )
            log_file.write('\n')
        log_file.write('\n')

        #********************************
        #*  run tensorflow sessions
        #********************************
        with tf.Session() as sess:
            # initialize the network
            print("Initialize the network...")
            if gb.RESUME:
                last_checkpoint = tf.train.latest_checkpoint(gb.EXP_ROOT)
                checkpoint_saver.restore(sess, last_checkpoint)
                print('Restoring from checkpoint: {}\n'.format(last_checkpoint))
                log_file.write('Restoring from checkpoint: {}\n'.format(last_checkpoint))
            else:
                sess.run(init_op)
                checkpoint_saver.save(sess, op.join( gb.EXP_ROOT, 'checkpoint'), global_step=0)

            # ********************************
            # *  run the training operation
            # ********************************
            start_step = sess.run(global_step)
            print('Starting training from iteration {}.\n'.format(start_step))
            log_file.write('Starting training from iteration {}.\n'.format(start_step))
            with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
                step=0
                flag_stop = False
                while(not flag_stop):
                    for b_feat_shape_, b_labels_shape_, b_sampleids_shape_, b_projViews_,\
                        b_feat_sketch_, b_labels_sketch_, b_sampleids_sketch_ in db_train.batches():
                        # run training
                        time_iter_start = time.time()
                        indx_shape = np.array( [B.shapeid2BIndxMap[p] for p in list(b_sampleids_shape_)] )
                        indx_sketch = np.array([B.sketchid2BIndxMap[p] for p in list(b_sampleids_sketch_ )])
                        _b_B_shape = B.B_shape[indx_shape,:]
                        _b_B_sketch = B.B_sketch[indx_sketch,:]
                        _b_B_shape = (_b_B_shape + 1) / 2
                        _b_B_sketch = (_b_B_sketch + 1) / 2
                        b_labels_shape_temp = np.expand_dims(b_labels_shape_, 1)
                        b_labels_shape_temp = np.tile(b_labels_shape_temp, [1, gb.BATCH_K*gb.BATCH_P])
                        b_labels_sketch_temp = np.expand_dims(b_labels_sketch_, 0)
                        b_labels_sketch_temp = np.tile(b_labels_sketch_temp, [gb.BATCH_K * gb.BATCH_P,1])
                        mask_ = np.array(np.equal(b_labels_shape_temp, b_labels_sketch_temp))
                        mask_ = mask_.astype("float32")
                        mask_ = 2*mask_-1
                        idx_neg = np.where( mask_<0 )
                        idx_pos = np.where(mask_ > 0)
                        mask_[idx_neg] = mask_[idx_neg]/(len(idx_neg[0]))
                        mask_[idx_pos] = mask_[idx_pos] / (len(idx_pos[0]))
                        mask_ = -mask_

                        feed_dict = {b_feat_shape: b_feat_shape_,
                                     b_label_shape: b_labels_shape_,
                                     b_feat_sketch: b_feat_sketch_,
                                     b_label_sketch: b_labels_sketch_,
                                     b_B_shape: _b_B_shape,
                                     b_B_sketch: _b_B_sketch,
                                     b_projViws: b_projViews_,
                                     mask: mask_}
                        _, _b_emb, _b_pool_weights, _b_labels_shape, b_val_loss, b_loss_shape, b_loss_sketch, step = \
                            sess.run( [train_op, net_shape['emb'], net_shape["pool_weights"], b_label_shape, val_loss, loss_shape, loss_sketch, global_step], feed_dict=feed_dict)
                        time_elapsed = time.time()-time_iter_start

                        # write summary and logging information
                        seconds_todo = (gb.MAX_NUM_ITER - step) * time_elapsed
                        print('iter{:6d}: loss={:.4f} | loss_shape={:.4f}  | loss_sketch={:.4f}   ETA: {} ({:.2f}s/it)'.format(
                                    step,
                                    float(b_val_loss),
                                    float(b_loss_shape),
                                    float(b_loss_sketch),
                                    timedelta(seconds=int(seconds_todo)),
                                    time_elapsed))
                        if( step%200==0 ):
                            log_file.write('iter:{:6d}, loss={:.4f} | loss_shape={:.4f}  | loss_sketch={:.4f}   ETA: {} ({:.2f}s/it)'.format(
                                    step,
                                    float(b_val_loss),
                                    float(b_loss_shape),
                                    float(b_loss_sketch),
                                    timedelta(seconds=int(seconds_todo)),
                                    time_elapsed))
                        if(step%gb.VALIDATION_FREQUENCY==0):
                            if gb.LOSS_FUNCTION == "SVP":
                                nn_test, ft_test, st_test, dcg_test, e_test, map_test, p_points_test, pre_test, rec_test, \
                                nn_train, ft_train, st_train, dcg_train, e_train, map_train, p_points_train, pre_train, rec_train,\
                                acc_test_sketch, acc_train_sketch, acc_shape, queryList, ranked_galleryList= \
                                    computeRetrievalMetrics(gb, sess, feat_shape, labels_shape, sampleids_shape, projViews, fids_shape,\
                                                            feat_sketch_train, labels_sketch_train, sampleids_sketch_train, fids_sketch_train,\
                                                            feat_sketch_test, labels_sketch_test, sampleids_sketch_test, fids_sketch_test)
                                print('\n***********************************************************\n'
                                      '     Test at iter:{:6d}, nn={:.3} ft={:.3} st={:.3}  e={:.3} dcg={:.3} map={:.3} \n'
                                      'Validation at iter:{:6d}, nn={:.3} ft={:.3} st={:.3} e={:.3} dcg={:.3} map={:.3} \n'
                                      'ACC at iter:{:6d}, test_sketch={:.3} train_sketch={:.3} shape={:.3}\n'
                                      '***********************************************************\n'.format(
                                    step, nn_test, ft_test, st_test, e_test, dcg_test, map_test,
                                    step, nn_train, ft_train, st_train, e_train, dcg_train, map_train,
                                    step, acc_test_sketch,acc_train_sketch, acc_shape ))
                                log_file.write('\n***********************************************************\n'
                                      '     Train at iter{:6d}: nn={:.3} ft={:.3} st={:.3} dcg={:.3} e={:.3} map={:.3} \n'
                                      'Validation at iter{:6d}: nn={:.3} ft={:.3} st={:.3} dcg={:.3} e={:.3} map={:.3} \n'
                                      'ACC at iter:{:6d}, test_sketch={:.3} train_sketch={:.3} shape={:.3}\n'
                                      '***********************************************************\n'.format(
                                    step, nn_test, ft_test, st_test, dcg_test, e_test, map_test,
                                    step, nn_train, ft_train, st_train, dcg_train, e_train, map_train,
                                    step, acc_test_sketch, acc_train_sketch, acc_shape))
                                log_file.write('Pre: \n')
                                for sss in range(len(pre_test)):
                                    log_file.write('{:.4} '.format(pre_test[sss]))
                                log_file.write('Rec: \n')
                                for sss in range(len(rec_test)):
                                    log_file.write('{:.4} '.format(rec_test[sss]))
                        # save trained models
                        if ( gb.CHECKPOINT_FREQUENCY > 0 and
                                step%gb.CHECKPOINT_FREQUENCY == 0):
                            checkpoint_saver.save(sess, os.path.join( gb.EXP_ROOT, 'checkpoint'), global_step=step)
                        if u.interrupted:
                            log_file.write("Interrupted on request!\n")
                            log_file.close()
                            checkpoint_saver.save(sess, op.join( gb.EXP_ROOT, 'checkpoint'), global_step=step)
                            break
                        if( step>=gb.MAX_NUM_ITER ):
                            flag_stop = True
                            break
    log_file.close()


if __name__=='__main__':
    main()