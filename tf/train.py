#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from random import shuffle
IMG_WIDTH = 64
IMG_HEIGHT = 44
IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT * 3

IMG_TYPE = "one" #one or two

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 60, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 19, 'Batch size *must devide the number of data evenly')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('glas_train_dir', '../image/'+IMG_TYPE+'/gla/train', 'Direcotry to put tha training data(gla)')
flags.DEFINE_string('nonglas_train_dir', '../image/'+IMG_TYPE+'/non_gla/train', 'Direcotry to put tha training data(nongla)')
flags.DEFINE_string('glas_test_dir', '../image/'+IMG_TYPE+'/gla/test', 'Direcotry to put tha test data(gla)')
flags.DEFINE_string('nonglas_test_dir', '../image/'+IMG_TYPE+'/non_gla/test', 'Direcotry to put tha test data(nongla)')
flags.DEFINE_string('train_dir', '../image/'+IMG_TYPE, 'directory to put summaries')
sess = tf.InteractiveSession()

def inferenceFromList(images_placeholder,  layers, keep_prob):

    #Initialize weight by norm(0, 0.1)
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    #Initialize bias by norm(0, 0.1)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #convolution layer
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')


    #pooling layer(2x2)
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Input Image
    x_image = tf.reshape(images_placeholder, [-1, IMG_HEIGHT, IMG_WIDTH, 3])

    convcnt = 0
    poolcnt = 0
    fccnt = 0
    
    lastLayer = [x_image]

    w_convs = []
    b_convs = []
    h_convs = []
    W_convs = []
    pools = []
    h_pools = []
    W_fcs = []
    b_fcs = []
    h_fcs = []    
    h_fcl_drops = []
    for layer in layers:
        print(layer.get("type"))
        if layer["type"] == "conv":
            convcnt += 1
            fromShape = layer["fromShape"]
            toChannel = layer["toChannel"]
            with tf.name_scope('conv%d'%convcnt) as scope:
                W_convs.append( weight_variable(fromShape + [toChannel]) )
                b_convs.append( bias_variable([toChannel]) )
                h_convs.append( tf.nn.relu(conv2d(lastLayer[-1], W_convs[-1]) + b_convs[-1]) )
                lastLayer.append(h_convs[-1])

        if layer["type"] == "pool":
            poolcnt += 1
            with tf.name_scope('pool%d'%poolcnt) as scope:
                h_pools.append(max_pool_2x2(lastLayer[-1]))
                lastLayer.append(h_pools[-1])
            
        if layer["type"] == "fc":
            fccnt += 1
            fromNodes = layer["fromNodes"]
            toNodes = layer["toNodes"]
            with tf.name_scope('fc%d'%fccnt) as scope:
                W_fcs.append(weight_variable([fromNodes, toNodes]))
                b_fcs.append( bias_variable([toNodes]))
                try:
                    tmp = tf.matmul(lastLayer[-1], W_fcs[-1]) + b_fcs[-1]
                    h_fcs.append(tmp)
                except :
                    lastLayer.append(tf.reshape(lastLayer[-1], [-1, fromNodes]))
                    tmp = tf.matmul(lastLayer[-1], W_fcs[-1]) + b_fcs[-1]
                    h_fcs.append(tmp)
                    
                lastLayer.append(h_fcs[-1])

                if "keep_prob" in layer:
                    h_fcl_drops.append(tf.nn.dropout(lastLayer[-1], keep_prob))
                    lastLayer.append(h_fcl_drops[-1])
            
                if "accelerate" in layer:
                    accelerateFunction = layer["accelerate"]
                    lastLayer.append(accelerateFunction(lastLayer[-1]))
                
    return lastLayer[-1]

def inference(images_placeholder, keep_prob):
    layers =[
        {
            "type": "conv",
            "fromShape":[5, 5, 3],
            "toChannel":32
        },
        
        {
            "type": "pool"
        },

        {
            "type": "conv",
            "fromShape":[5, 5, 32],
            "toChannel":64
        },

        {
            "type": "pool"
        },

        {
            "type": "fc",
            "fromNodes":IMG_HEIGHT*IMG_WIDTH*4,
            "toNodes": 1024,
            "accelerate": tf.nn.relu,
            "keep_prob": 1
        },

        {
            "type": "fc",
            "fromNodes":1024,
            "toNodes":2,
            "accelerate": tf.nn.softmax
        }
    ]
    return inferenceFromList(images_placeholder, layers, keep_prob)
            

def _inference(images_placeholder, keep_prob):

    #Initialize weight by norm(0, 0.1)
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    #Initialize bias by norm(0, 0.1)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #convolution layer
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')


    #pooling layer(2x2)
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Input Image
    x_image = tf.reshape(images_placeholder, [-1, IMG_HEIGHT, IMG_WIDTH, 3])


    #convolution layer1
    #5x5patch(3channel(RGB)) -> 32channel
    #activation function :ReLU
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    #pooling layer1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    #convolution layer2
    #5x5patch(32channel) -> 64channel
    #activation function :ReLU
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    #pooling layer2
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    #fully connected layer1
    #compressed IMG -> 1024 node layer
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([int(IMG_HEIGHT*IMG_WIDTH*64/16), 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, int(IMG_HEIGHT*IMG_WIDTH*64/16)])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)

    #fully_connected layer2
    #1024 nodes -> 2 nodes (Yes No)
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

    #softmax
    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)
        ##y_conv = tf.matmul(h_fcl_drop, W_fc2) + b_fc2

    return y_conv





def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    tf.scalar_summary("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    loga = tf.argmax(logits, 1)
    laba = tf.argmax(labels, 1)
    oo = tf.reduce_sum(loga * laba)
    labas = tf.reduce_sum(laba)
    logas = tf.reduce_sum(loga)
    f_measure = 2*oo/(labas+logas)
    tf.scalar_summary("F_measure", f_measure)
    return f_measure




if __name__ == '__main__':

    train_image = []
    train_label = []
   
    #trainning data (gla)
    files = os.listdir(FLAGS.glas_train_dir)
    for file in files:
        img = cv2.imread(FLAGS.glas_train_dir + "/" + file)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        train_image.append(img.flatten().astype(np.float32)/255.0)

        tmp = np.zeros(2)
        tmp[1] = 1
        train_label.append(tmp)

    #trainning data (nongla)
    files = os.listdir(FLAGS.nonglas_train_dir)
    for file in files:
        img = cv2.imread(FLAGS.nonglas_train_dir + "/" + file)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        train_image.append(img.flatten().astype(np.float32)/255.0)

        tmp = np.zeros(2)
        tmp[0] = 1
        train_label.append(tmp)

    pairs = [ i for i in zip(train_image, train_label)]
    shuffle(pairs)
    train_image, train_label = [i for i in zip(*pairs)]
        
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)





    
    test_image = []
    test_label = []
    
    #test data (gla)
    files = os.listdir(FLAGS.glas_test_dir)
    for file in files:
        img = cv2.imread(FLAGS.glas_test_dir + "/" + file)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        test_image.append(img.flatten().astype(np.float32)/255.0)

        tmp = np.zeros(2)
        tmp[1] = 1
        test_label.append(tmp)


    #test data (nongla)
    files = os.listdir(FLAGS.nonglas_test_dir)
    for file in files:
        img = cv2.imread(FLAGS.nonglas_test_dir + "/" + file)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

        test_image.append(img.flatten().astype(np.float32)/255.0)

        tmp = np.zeros(2)
        tmp[0] = 1
        test_label.append(tmp)

    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)


    
    with tf.Graph().as_default():

        #import IMG
        images_placeholder = tf.placeholder("float", shape=(None, IMG_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, 2))
        keep_prob = tf.placeholder("float")

        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, FLAGS.learning_rate)
        acc=accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        hoge = logits

        #Train!!
        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_image)/FLAGS.batch_size)):
                batch = FLAGS.batch_size*i

                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.6})

            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))

            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})

            summary_writer.add_summary(summary_str, step)
            """
            print(sess.run(hoge, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0}))
            """

            print("test accuracy %g"%sess.run(acc, feed_dict={
                images_placeholder: test_image,
                labels_placeholder: test_label,
                keep_prob: 1.0}))

            
    print("test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    """
    print(sess.run(hoge, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    print(sess.run(hoge, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0}))
    """
    save_path = saver.save(sess, "model.ckpt")
    


