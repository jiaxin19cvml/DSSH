import tensorflow as tf
from models import loss_func

slim=tf.contrib.slim

def shapeHashLayers(input, emb_dim, is_training, is_reuse, name=None):
    if not name==None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net = _view_pool( net, 'view_pooling' )
            net['fc1'] = slim.fully_connected(
                net['view_pooling'], 1024, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc1')
            net['fc2'] = slim.fully_connected(
                net['fc1'], 512, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc2')
            net['fc3'] = slim.fully_connected(
                net['fc2'], 256, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc3')
            net['emb'] = slim.fully_connected(
                net['fc3'], emb_dim, activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc4')
        return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net = _view_pool(net, 'view_pooling')
        net['fc1'] = slim.fully_connected(
            net['view_pooling'], 1024, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc1')
        net['fc2'] = slim.fully_connected(
            net['fc1'], 512, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc2')
        net['fc3'] = slim.fully_connected(
            net['fc2'], 256, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc3')
        net['emb']  = slim.fully_connected(
            net['fc3'], emb_dim, activation_fn=None,
            weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc4')
    return net

def sketchHashLayers(input, emb_dim, is_training, is_reuse, name=None):
    if not name==None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net['fc1'] = slim.fully_connected(
                net['input'], 1024, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc1')
            net['fc2'] = slim.fully_connected(
                net['fc1'], 512, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc2')
            net['fc3'] = slim.fully_connected(
                net['fc2'], 256, normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'decay': 0.9,
                    'epsilon': 1e-5,
                    'scale': True,
                    'is_training': is_training,
                    # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
                }, reuse=is_reuse, scope='fc3')
            net['emb']  = slim.fully_connected(
                net['fc3'], emb_dim, activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc4')
        return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net['fc1'] = slim.fully_connected(
            net['input'], 1024, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc1')
        net['fc2'] = slim.fully_connected(
            net['fc1'], 512, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc2')
        net['fc3'] = slim.fully_connected(
            net['fc2'], 256, normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': is_training,
                # 'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }, reuse=is_reuse, scope='fc3')
        net['emb']  = slim.fully_connected(
            net['fc3'], emb_dim, activation_fn=None,
            weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc4')
    return net


def shapeShallowHashLayers(input, emb_dim, is_training, is_reuse, name=None):
    if not name==None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net = _view_pool(net, 'view_pooling')
            net['fc1'] = slim.fully_connected(
                net['view_pooling'], 1024, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc1')
            net['fc2'] = slim.fully_connected(
                net['fc1'], 512, activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc2')
            net['emb'] = slim.fully_connected( net['fc2'], emb_dim, activation_fn=tf.nn.tanh,
            weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc3')

            return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net = _view_pool(net, 'view_pooling')
        net['fc1'] = slim.fully_connected(
            net['view_pooling'], 1024, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc1')
        net['fc2'] = slim.fully_connected(
            net['fc1'], 512, activation_fn=None,
            weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc2')
        net['emb'] = slim.fully_connected(net['fc2'], emb_dim, activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc3')

        return net

def sketchShallowHashLayers(input, emb_dim, is_training, is_reuse, name=None):
    if not name==None:
        with tf.variable_scope(name):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net['input'] = input
            net['fc1'] = slim.fully_connected(
                net['input'], 1024, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc1')

            net['emb'] = slim.fully_connected(
                net['fc1'], emb_dim, activation_fn=tf.nn.tanh,
                weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc2')
            return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net['input'] = input
        net['fc1'] = slim.fully_connected(
            net['input'], 1024, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc1')

        net['emb'] = slim.fully_connected(
            net['fc1'], emb_dim, activation_fn=tf.nn.tanh,
            weights_initializer=tf.orthogonal_initializer(), reuse=is_reuse, scope='fc2')
        return net

def view_pool(input, name):
    pooled_input = tf.reduce_mean( input, [1], name=name )

    return pooled_input

def view_exitation( view, is_reuse ):
    hidden_pool = slim.fully_connected(
        view, 100, activation_fn=tf.nn.relu, reuse=is_reuse, scope='hidden_pool')
    out_pool = slim.fully_connected(
        hidden_pool, 1, activation_fn=None, reuse=is_reuse, scope='output_pool')

    return out_pool

def attention_pool(input_, projViews, numb_channles, is_reuse, name):
    projViews = tf.transpose(projViews, perm=[1, 0, 2])
    input = tf.transpose(input_, perm=[1, 0, 2])
    hybrid = tf.concat([input, projViews], 2)
    for i in range(numb_channles):
        if is_reuse:
            is_reuse_exitation = True
        else:
            is_reuse_exitation = (i != 0)
        view = tf.gather(hybrid, i)
        single_view_weight = view_exitation( view, is_reuse_exitation )
        #if i==0:
         #   view_weights = tf.expand_dims(single_view_weight, 0)
        #else:
        #    view_weights = tf.concat([view_weights, tf.expand_dims(single_view_weight, 0)], 0)

        if i == 0:
            view_weights_ = single_view_weight
        else:
            view_weights_ = tf.concat([view_weights_, single_view_weight], 1)

    #view_weights = slim.fully_connected(
    #        view_weights_,numb_channles, activation_fn=tf.nn.softmax, reuse=is_reuse, scope='comb')
    view_weights = view_weights_
    #input = tf.transpose(input_, perm=[1, 0, 2])
    #output = tf.reduce_max( tf.multiply( input,view_weights ), [0] )
    #output = tf.reduce_max(tf.multiply(input, view_weights), [0])
    view_weights = tf.expand_dims(view_weights, 2)
    view_weights = tf.tile( view_weights, [1,1,1536] )
    #view_weights = tf.tile(view_weights, [1, 1, 2048])
    output = tf.multiply( input_,view_weights )
    #output = tf.transpose(input, perm=[1, 0, 2])
    #output = input_

    return output, view_weights

def siameseHashLayers(net, emb_dim, is_reuse):

    net['fc1'] = slim.fully_connected(
        net['input'], 1024, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc1')
    net['fc2'] = slim.fully_connected(
        net['fc1'], 512, activation_fn=tf.nn.relu, reuse=is_reuse, scope='fc2')
    net['emb'] = slim.fully_connected( net['fc2'], emb_dim, activation_fn=tf.nn.tanh, reuse=is_reuse, scope='fc3')

    return net


def _view_pool(view_features, name):
    vp = tf.expand_dims(view_features[0], 0)
    for v in view_features[1:]:
        v=tf.expand_dims(v, 0)
        vp=tf.concat([vp,v], 0, name=name)
    vp = tf.reduce_mean(vp, [0], name=name)
    return vp

def shapeHashNet(feat, projViews, gb, is_training, is_reuse):
    if not gb.NAME_SUBNET_SHAPE==None:
        with tf.variable_scope(gb.NAME_SUBNET_SHAPE):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            #net["input"] = view_pool(feat, "view_pooling")
            weighted_input, pool_weights = attention_pool(feat, projViews, gb.NUMB_CHANNELS, is_reuse, "view_pooling")
            net["pool_weights"] = pool_weights

            input = tf.transpose(weighted_input, perm=[1, 0, 2])
            emb_pool = []
            for i in range(gb.NUMB_CHANNELS):
                if is_reuse:
                    is_reuse = True
                else:
                    is_reuse = (i != 0)
                view = tf.gather(input, i)
                net["input"] = view
                net = siameseHashLayers( net, gb.EMD_DIM, is_reuse )
                if i==0:
                    emb_pool = tf.expand_dims(net["emb"], 0)
                else:
                    v = tf.expand_dims(net["emb"], 0)
                    emb_pool = tf.concat([emb_pool, v], 0 )
                net["emb_vp"] = tf.reduce_mean(emb_pool, [0], name="emb_vp")

            return net
    else:
        net = slim.utils.convert_collection_to_dict(('none', 1))
        # net["input"] = view_pool(feat, "view_pooling")
        weighted_input, pool_weights = attention_pool(feat, projViews, gb.NUMB_CHANNELS, is_reuse, "view_pooling")
        net["pool_weights"] = pool_weights

        input = tf.transpose(weighted_input, perm=[1, 0, 2])
        emb_pool = []
        for i in range(gb.NUMB_CHANNELS):
            if is_reuse:
                is_reuse = True
            else:
                is_reuse = (i != 0)
            view = tf.gather(input, i)
            net["input"] = input
            net = siameseHashLayers(net, gb.EMD_DIM, is_reuse)
            if i == 0:
                emb_pool = tf.expand_dims(net["emb"], 0)
            else:
                v = tf.expand_dims(net["emb"], 0)
                emb_pool = tf.concat([emb_pool, v], 0)
            net["emb_vp"] = vp = tf.reduce_mean(vp, [0], name="emb_vp")

        return net

def sketchHashNet(feat, gb, is_training, is_reuse):
    if not gb.NAME_SUBNET_SKETCH==None:
        if gb.NAME_SUBNET_SHAPE == gb.NAME_SUBNET_SKETCH:
            is_reuse = True
        with tf.variable_scope(gb.NAME_SUBNET_SKETCH):
            net = slim.utils.convert_collection_to_dict(('none', 1))
            net["input"] = feat
            net = siameseHashLayers( net, gb.EMD_DIM, is_reuse )

            return net
    else:
        if gb.NAME_SUBNET_SHAPE == gb.NAME_SUBNET_SKETCH:
            is_reuse = True
        net = slim.utils.convert_collection_to_dict(('none', 1))
        net["input"] = feat
        if gb.NAME_SUBNET_SHAPE == gb.NAME_SUBNET_SKETCH:
            is_reuse = True
        net = siameseHashLayers(net, gb.EMD_DIM, is_reuse)

        return net


def loss(emb_shape, emb_sketch, labels_shape, labels_sketch, gb):
    # set parameters for metric learning with triplet loss
    tripLoss_choice=slim.utils.convert_collection_to_dict(('none', 1))
    tripLoss_choice['triplet_loss'] = 'batch_hard'
    tripLoss_choice['margin'] = 'soft'
    tripLoss_choice['metric']  = 'euclidean'
    tripLoss_choice['batch_k'] = gb.BATCH_K

    # calculate the distance for metric learning with triplet loss
    dists_shape = loss_func.cdist(emb_shape, emb_shape, metric=tripLoss_choice['metric'])
    dists_sketch = loss_func.cdist(emb_sketch, emb_sketch, metric=tripLoss_choice['metric'])

    emb_comb = tf.concat([emb_shape, emb_sketch], axis=0)
    labels_comb = tf.concat([labels_shape, labels_sketch], axis=0)
    dists_comb = loss_func.cdist(emb_comb, emb_comb, metric=tripLoss_choice['metric'])

    # calculate the triplet loss for metric learning
    triplet_loss_shape, train_top1_shape, prec_at_k_shape, _, neg_dists_shape, pos_dists_shape = loss_func.LOSS_CHOICES[tripLoss_choice['triplet_loss']](
        dists_shape, labels_shape, tripLoss_choice['margin'], batch_precision_at_k=tripLoss_choice['batch_k']-1)
    triplet_loss_sketch, train_top1_sketch, prec_at_k_sketch, _, neg_dists_sketch, pos_dists_sketch = loss_func.LOSS_CHOICES[
        tripLoss_choice['triplet_loss']](
        dists_sketch, labels_sketch, tripLoss_choice['margin'], batch_precision_at_k=tripLoss_choice['batch_k'] - 1)
    triplet_loss_comb, train_top1_comb, prec_at_k_comb, _, neg_dists_comb, pos_dists_comb = loss_func.LOSS_CHOICES[
        tripLoss_choice['triplet_loss']](
        dists_comb, labels_comb, tripLoss_choice['margin'], batch_precision_at_k=tripLoss_choice['batch_k']*2 - 1)

    crossModality_loss = loss_func.reduceXMean(emb_shape, emb_sketch, labels_shape, BATCH_P, BATCH_K)
    mean_triplet_loss_all = (tf.reduce_mean(triplet_loss_shape)+tf.reduce_mean(triplet_loss_sketch)+tf.reduce_mean(triplet_loss_comb))/3.0
    val_loss=crossModality_loss+mean_triplet_loss_all

    return val_loss, triplet_loss_shape, triplet_loss_sketch, triplet_loss_comb, crossModality_loss, prec_at_k_shape, prec_at_k_sketch, prec_at_k_comb

def loss_triplet( emb, labels, gb ):
    # set parameters for metric learning with triplet loss
    tripLoss_choice=slim.utils.convert_collection_to_dict(('none', 1))
    tripLoss_choice['triplet_loss'] = 'batch_hard'
    tripLoss_choice['margin'] = 'soft'
    tripLoss_choice['metric']  = 'euclidean'
    tripLoss_choice['batch_k'] = gb.BATCH_K

    # calculate the distance for metric learning with triplet loss
    dists = loss_func.cdist(emb, emb, metric=tripLoss_choice['metric'])

    # calculate the triplet loss for metric learning
    triplet_loss, train_top1_shape, prec_at_k_shape, _, neg_dists, pos_dists = loss_func.LOSS_CHOICES[tripLoss_choice['triplet_loss']](
        dists, labels, tripLoss_choice['margin'], batch_precision_at_k=tripLoss_choice['batch_k']-1)

    val_loss = triplet_loss
    val_loss_mean = tf.reduce_mean(val_loss)

    return val_loss_mean

def loss_crossentropy(gb, emb, labels):
    #if len(labels.shape)==1:
    #    labels = tf.one_hot(labels, gb.NUMB_CLASS)
    val_loss = tf.losses.softmax_cross_entropy(labels, emb, label_smoothing=0)

    return val_loss

def loss_sigmoid_crossentropy(gb, emb, labels):
    #if len(labels.shape)==1:
    #    labels = tf.one_hot(labels, gb.NUMB_CLASS)
    val_loss = tf.losses.sigmoid_cross_entropy(labels, emb, label_smoothing=0)

    return val_loss

def loss_SVP(emb_shape, emb_sketch, labels_shape, labels_sketch, b_shape, b_sketch, gb):

    # set parameters for metric learning with triplet loss
    #labels_shape = tf.one_hot( labels_shape, gb.EMD_DIM)
    #labels_sketch = tf.one_hot(labels_sketch, gb.EMD_DIM)
    loss_reghash_shape = loss_sigmoid_crossentropy(gb, emb_shape, b_shape )
    loss_reghash_sketch = loss_sigmoid_crossentropy(gb, emb_sketch, b_sketch )


    emb_comb = tf.concat([emb_shape, emb_sketch], axis=0)
    labels_comb = tf.concat([labels_shape, labels_sketch], axis=0)
    loss_metric = loss_triplet( emb_comb,labels_comb, gb )

    val_loss = loss_metric#(loss_reghash_shape+loss_reghash_sketch)+20*loss_metric

    return val_loss, loss_reghash_shape, loss_reghash_sketch

def cross_entropy(a,b):
    val = tf.reduce_mean(-tf.reduce_sum(a * tf.log(b)+(1-a)*tf.log(1-b), reduction_indices=[1]))
    return val

def loss_SVP_siamese(emb_shape, emb_sketch, labels_shape, labels_sketch, b_shape, b_sketch, b_feat_shape_pooled, b_feat_sketch, mask, gb):


    emb_comb = tf.concat([emb_shape, emb_sketch], axis=0)
    labels_comb = tf.concat([labels_shape, labels_sketch], axis=0)
    loss_metric = loss_triplet( emb_comb,labels_comb, gb )

    loss_cor = tf.multiply( tf.sigmoid(tf.matmul( emb_shape, tf.transpose(emb_sketch,[1,0]) )), mask)
    loss_cor = tf.reduce_sum( loss_cor )
    #loss_cor = tf.losses.sigmoid_cross_entropy(emb_shape,emb_sketch )

    #feat_comb = tf.concat([b_feat_shape_pooled, b_feat_sketch], axis=0)

    #loss_metric_feat = loss_triplet(feat_comb, labels_comb, gb)

    val_loss = loss_metric+1*loss_cor#+loss_metric_feat#(loss_reghash_shape+loss_reghash_sketch)+5*loss_metric
    loss_metric_feat=loss_cor
    loss_reghash_sketch=val_loss

    return val_loss, loss_metric_feat, loss_reghash_sketch

