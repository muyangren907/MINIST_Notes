#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/6/28/0028 13:39
# @Author  : muyangren907
# @FileName: MNISTtest.py
# @Software: PyCharm
from PIL import Image
import numpy as np
import tensorflow as tf
import input_data
import os

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 执行read_data_sets()函数将会返回一个DataSet实例，其中包含了下面三个数据集。

train = mnist.train  # 55000 组 图片和标签, 用于训练。
validation = mnist.validation  # 5000 组 图片和标签, 用于迭代验证训练的准确性。
test = mnist.test  # 10000 组 图片和标签, 用于最终测试训练的准确性。


def dataset2pic(dataset, savepath, colmax=50, rowmax=50):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imgdatas = dataset.images
    arrcol = np.zeros([28, 28])
    arrrow = np.zeros([28, 280])
    colindex, rowindex = 0, 0
    one = np.ones([28 * colmax, 28 * rowmax]) * 255
    # colmax, rowmax = 50, 50
    for imgindex in range(len(imgdatas)):
        imagedata = imgdatas[imgindex]
        k = len(imagedata) // 28
        arr = np.array([imagedata[0:28]])
        for i in range(1, k):
            arr = np.vstack([arr, imagedata[i * 28:(i + 1) * 28]])

        # print(len(arrcol))
        # print(len(arr))
        if colindex == 0:
            arrcol = arr
            colindex += 1
        else:
            arrcol = np.column_stack((arrcol, arr))
            colindex += 1
            if colindex == colmax:
                colindex = 0
                if rowindex == 0:
                    arrrow = arrcol
                    rowindex += 1
                else:
                    arrrow = np.row_stack((arrrow, arrcol))
                    rowindex += 1
                    if rowindex == rowmax:
                        rowindex = 0
                        imgid = ((imgindex + 1) // (rowmax * colmax))
                        imgname = '%03d' % imgid
                        img = Image.fromarray(arrrow * 255).convert('RGB')
                        img.save('%s/%s_1.png' % (savepath, imgname))
                        img = Image.fromarray(one - arrrow * 255).convert('RGB')
                        img.save('%s/%s_2.png' % (savepath, imgname))
                        print('已成功输出%s\t%s' % (savepath, imgid))

        # img = Image.fromarray(arr * 255).convert('RGB')
        # img.save('%s/%05d.png' % (savepath, imgindex + 1))
        # print('%s成功保存至%s' % (imgindex + 1, savepath))


def testfun():
    dataset2pic(train, 'images1/train')
    dataset2pic(validation, 'images1/validation')
    dataset2pic(test, 'images1/test')
    # testimages, testlabels = train.images, train.labels
    #
    # imagedata = testimages[0]
    # # print(imagedata)
    # k = len(imagedata) // 28
    # print(k)
    # arr = np.array([imagedata[0:28]])
    # for i in range(1, k):
    #     arr = np.vstack([arr, imagedata[i * 28:(i + 1) * 28]])
    #     # print(imagedata[i * 28:(i + 1) * 28])
    # print(arr)
    # # print(testimages[0]*255, testlabels[0])
    # # data = testimages[0]
    # new_im = Image.fromarray(arr * 255)
    # new_im = new_im.convert('RGB')
    # new_im.show()
    # new_im.save('images/test/0.png')


def MINISbasic():
    x = tf.compat.v1.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.compat.v1.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y))

    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 初始化我们创建的变量
    # init = tf.initialize_all_variables()
    init = tf.compat.v1.global_variables_initializer()

    # 现在我们可以在一个Session里面启动我们的模型，并且初始化变量
    # sess = tf.Session()
    sess = tf.compat.v1.Session()
    sess.run(init)

    # 然后开始训练模型，这里我们让模型循环训练1000次！
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 最后，我们计算所学习到的模型在测试数据集上面的正确率。
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def MINISadvanced():
    # sess = tf.InteractiveSession()
    sess = tf.compat.v1.InteractiveSession()

    # 通过为输入图像和目标输出类别创建节点，来开始构建计算图。
    x = tf.compat.v1.placeholder("float", shape=[None, 784])
    y_ = tf.compat.v1.placeholder("float", shape=[None, 10])

    # 一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改。在机器学习的应用过程中，模型参数一般用Variable来表示。
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.compat.v1.global_variables_initializer())
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y))

    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    def weight_variable(shape):
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.compat.v1.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y_conv))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    # print(validation)
    # images_feed, labels_feed = train.next_batch(100)
    # print(images_feed)
    # print(labels_feed)
    # print(len(train.images[0]))
    # image0 = train.images[0]
    # for i in range(784):
    #     if i % 28 == 0:
    #         print()
    #     print(image0[i], end=' '*10)

    # labels = train.labels
    # print(labels[0])
    # for i in range(10):
    #     if labels[0][i] == 1:
    #         print(i)

    # x = tf.placeholder("float", [None, 784])

    # MINISbasic()

    # MINISadvanced()
    testfun()
