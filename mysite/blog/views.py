from PIL import Image
import tensorflow as tf
from numpy import *
from time import sleep
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile
from .models import Post
from django.shortcuts import render_to_response
from django.shortcuts import render_to_response
from django.template import RequestContext
from .models import TestData
import json


# Create your views here.
@csrf_exempt
def post_list(request):
    # temp = 0
    # result = list(request.POST.keys())
    # if(request.method == 'POST'):
    #     data = result[0][22:]
    #     img = base64.b64decode(data)
    #
    #     fh = open("imageToSave.png", "wb")
    #     fh.write(img)
    #     fh.close()
    #     print("save success!")
    #
    #     im = Image.open("imageToSave.png")
    #     rgb_im = im.convert('RGB')
    #     rgb_im.save('colors.jpg')
    #
    #
    #     im=Image.open('colors.jpg')
    #     img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
    #     data = img.reshape([1, 784])
    #     data = 1 - (data/255)
    #
    #
    #
    #     tf.set_random_seed(777)  # reproducibility
    #
    #     # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #     # Check out https://www.tensorflow.org/get_started/mnist/beginners for
    #     # more information about the mnist dataset
    #
    #     # hyper parameters
    #     learning_rate = 0.001
    #     training_epochs = 15
    #     batch_size = 100
    #
    #     # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
    #     keep_prob = tf.placeholder(tf.float32)
    #
    #     # input place holders
    #     X = tf.placeholder(tf.float32, [None, 784])
    #     X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
    #     Y = tf.placeholder(tf.float32, [None, 10])
    #
    #     # L1 ImgIn shape=(?, 28, 28, 1)
    #     W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    #     #    Conv     -> (?, 28, 28, 32)
    #     #    Pool     -> (?, 14, 14, 32)
    #     L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    #     L1 = tf.nn.relu(L1)
    #     L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
    #                         strides=[1, 2, 2, 1], padding='SAME')
    #     L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    #     '''
    #     Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
    #     Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
    #     Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
    #     Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
    #     '''
    #
    #     # L2 ImgIn shape=(?, 14, 14, 32)
    #     W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #     #    Conv      ->(?, 14, 14, 64)
    #     #    Pool      ->(?, 7, 7, 64)
    #     L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #     L2 = tf.nn.relu(L2)
    #     L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
    #                         strides=[1, 2, 2, 1], padding='SAME')
    #     L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    #     '''
    #     Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
    #     Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
    #     Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
    #     Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
    #     '''
    #
    #     # L3 ImgIn shape=(?, 7, 7, 64)
    #     W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #     #    Conv      ->(?, 7, 7, 128)
    #     #    Pool      ->(?, 4, 4, 128)
    #     #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
    #     L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    #     L3 = tf.nn.relu(L3)
    #     L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
    #                         1, 2, 2, 1], padding='SAME')
    #     L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    #     L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
    #     '''
    #     Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
    #     Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
    #     Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
    #     Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
    #     Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
    #     '''
    #
    #     # L4 FC 4x4x128 inputs -> 625 outputs
    #     W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
    #                          initializer=tf.contrib.layers.xavier_initializer())
    #     b4 = tf.Variable(tf.random_normal([625]))
    #     L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    #     L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    #     '''
    #     Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    #     Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    #     '''
    #
    #     # L5 Final FC 625 inputs -> 10 outputs
    #     W5 = tf.get_variable("W5", shape=[625, 10],
    #                          initializer=tf.contrib.layers.xavier_initializer())
    #     b5 = tf.Variable(tf.random_normal([10]))
    #     logits = tf.matmul(L4, W5) + b5
    #     saver = tf.train.Saver()
    #     init_op = tf.global_variables_initializer()
    #     with tf.Session() as sess:
    #         sess.run(init_op)
    #         save_path = "./minist_softmax.ckpt"
    #         saver.restore(sess, save_path)
    #         prediction = sess.run(
    #         tf.argmax(logits, 1), feed_dict={X: data, keep_prob: 1})
    #         temp = prediction
    return render(request, 'blog/post_list.html', {})

@csrf_exempt
def data_return(request):
    temp = 0
    result = list(request.POST.keys())
    if(request.method == 'POST'):
        data = result[0][22:]
        img = base64.b64decode(data)

        fh = open("imageToSave.png", "wb")
        fh.write(img)
        fh.close()
        print("save success!")

        im = Image.open("imageToSave.png")
        rgb_im = im.convert('RGB')
        rgb_im.save('colors.jpg')


        im=Image.open('colors.jpg')
        img = array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
        data = img.reshape([1, 784])
        data = 1 - (data/255)



        tf.set_random_seed(777)  # reproducibility

        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # Check out https://www.tensorflow.org/get_started/mnist/beginners for
        # more information about the mnist dataset

        # hyper parameters
        learning_rate = 0.001
        training_epochs = 15
        batch_size = 100

        # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
        keep_prob = tf.placeholder(tf.float32)

        # input place holders
        X = tf.placeholder(tf.float32, [None, 784])
        X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
        Y = tf.placeholder(tf.float32, [None, 10])

        # L1 ImgIn shape=(?, 28, 28, 1)
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
        '''
        Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
        Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
        Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
        '''

        # L2 ImgIn shape=(?, 14, 14, 32)
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
        '''
        Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
        Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
        Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
        '''

        # L3 ImgIn shape=(?, 7, 7, 64)
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                            1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
        L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
        '''
        Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
        Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
        Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
        '''

        # L4 FC 4x4x128 inputs -> 625 outputs
        W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([625]))
        L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
        '''
        Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
        Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
        '''

        # L5 Final FC 625 inputs -> 10 outputs
        W5 = tf.get_variable("W5", shape=[625, 10],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))
        logits = tf.matmul(L4, W5) + b5
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            save_path = "./minist_softmax.ckpt"
            saver.restore(sess, save_path)
            prediction = sess.run(
            tf.argmax(logits, 1), feed_dict={X: data, keep_prob: 1})
            temp = prediction
    # return render(request, 'blog/post_list.html', {})
    return render(request,'blog/test.html',{'tests' : temp})
