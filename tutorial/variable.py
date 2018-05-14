# !/usr/bin/env python2
# -*- coding: utf-8 -*-
# blog： http://blog.csdn.net/jerr__y/article/details/70809528
# github： https://github.com/yongyehuang/Tensorflow-Tutorial


import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def test1():
    # 1. tf.placeholder
    print('---------- tf.placeholder ---------- ')
    v1 = tf.placeholder(tf.float32, shape=[2, 3, 4])
    print(v1.name)
    v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
    print(v1.name)
    v1 = tf.placeholder(tf.float32, shape=[2, 3, 4], name='ph')
    print(v1.name)
    print(type(v1))
    print(v1)

    # 2. tf.Variable
    print('\n---------- tf.Variable ---------- ')
    v2 = tf.Variable([1, 2], dtype=tf.float32)
    print(v2.name)
    v2 = tf.Variable([1, 2], dtype=tf.float32, name='Var')
    print(v2.name)
    v2 = tf.Variable([1, 2], dtype=tf.float32, name='Var')
    print(v2.name)
    print(type(v2))
    print(v2)

    # 3. tf.get_variable
    print('\n---------- tf.get_variable ---------- ')
    v3 = tf.get_variable(name='gv', shape=[])
    print(v3.name)
    # 如果name重名会抛出异常
    # v4 = tf.get_variable(name='gv', shape=[2])
    # print(v4.name)
    print(type(v3))
    print(v3)

    # 4.
    print('\n---------- get all variable that trainable is True ---------- ')
    vs = tf.trainable_variables()
    print(len(vs))
    for v in vs:
        print(v)

    '''
        结论：
        第4步打印出4条记录，其中前3个是tf.Variable定义的，第4个是tr.get_variable。因为placeholder的trainable为False，所以没有输出。
        tf.placeholder() 占位符。* trainable==False *
        tf.Variable() 一般变量用这种方式定义。 * 可以选择 trainable 类型 *
        tf.get_variable() 一般都是和 tf.variable_scope() 配合使用，从而实现变量共享的功能。 * 可以选择 trainable 类型 *
    '''


def test2():
    with tf.name_scope('nsc1'):
        v1 = tf.Variable([1], name='v1')
        with tf.variable_scope('vsc1'):
            v2 = tf.Variable([1], name='v2')
            v3 = tf.get_variable(name='gv1', shape=[])
        with tf.name_scope('nsc2'):
            v4 = tf.Variable([1], name='v4')
            v5 = tf.get_variable(name='gv2', shape=[])

    print('v1.name: ', v1.name)
    print('v2.name: ', v2.name)
    print('v3.name: ', v3.name)
    print('v4.name: ', v4.name)
    print('v5.name: ', v5.name)

    '''
    输出结果，注意tf.get_variable定义的变量：
        ('v1.name: ', u'nsc1/v1:0')
        ('v2.name: ', u'nsc1/vsc1/v2:0')
        ('v3.name: ', u'vsc1/gv1:0')        gv1在variable_scope('vsc1')下，所以name多个vsc1的前缀。和name_scope没有任何关系。
        ('v4.name: ', u'nsc1/nsc2/v4:0')
        ('v5.name: ', u'gv2:0')             gv2在name_scope('nsc2')下，对gv2的name没有任何影响。
        tf.name_scope() 并不会对 tf.get_variable() 创建的变量有任何影响。 
        tf.name_scope() 主要是用来管理命名空间的，这样子让我们的整个模型更加有条理。
        而 tf.variable_scope() 的作用是为了实现变量共享，它和 tf.get_variable() 来完成变量共享的功能。
    '''


def test3():
    # with tf.name_scope('nsc1') as nsc1:
    #     print tf.get_variable_scope().original_name_scope
    #     with tf.variable_scope(tf.get_variable_scope()) as vsc1:
    #         print vsc1
    #         print vsc1.name

    with tf.name_scope('nsc1'):
        with tf.name_scope('nsc2') as nsc2:
            print nsc2  # 输出：nsc1/nsc2/，说明name_scope会嵌套

    print
    with tf.variable_scope('vsc1'):
        with tf.variable_scope('vsc2') as vsc2:
            print vsc2
            print vsc2.name  # vsc1/vsc2，说明variable_scope会嵌套
            print vsc2.original_name_scope
            print tf.get_variable_scope().original_name_scope

    print
    with tf.variable_scope("hello") as variable_scope:
        arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)

        print variable_scope
        print variable_scope.name  # 打印出变量空间名字
        print arr1.name  # 输出hello/arr1:0
        print tf.get_variable_scope().original_name_scope
        # tf.get_variable_scope() 获取的就是variable_scope

        with tf.variable_scope("xixi") as v_scope2:
            print tf.get_variable_scope().original_name_scope  # 输出：hello/xixi/
            # tf.get_variable_scope() 获取的就是v _scope2
            arr2 = tf.get_variable("arr2", shape=[1])
            print arr2.name  # 输出：hello/xixi/arr2:0

        with tf.variable_scope(tf.get_variable_scope()) as v_scope3:
            print v_scope3.name                     # hello
            print v_scope3.original_name_scope      # hello/
            arr3 = tf.get_variable("arr3", shape=[1])
            print arr3.name                         # hello/arr3:0
            '''
            看到很多代码用tf.variable_scope(f.get_variable_scope())，这种方式是说上一层定义了tf.variable_scope, 用上一层的scope, 
            保证还是在同一个scope下, 没有再嵌套.
            '''

        with tf.variable_scope("hello") as v_scope4:
            print v_scope4.name                     # hello/hello
            print v_scope4.original_name_scope      # hello/hello_1/
            '''
            用字符串定义之前同名的scope，显然还是要嵌套的
            '''

    with tf.name_scope("name1"):
        with tf.variable_scope("var1"):
            w = tf.get_variable("w", shape=[2])
            res = tf.add(w, [3])
            print w.name        # var1/w:0
            print res.name      # name1/var1/Add:0, name_scope、variable_scope对op都生效


def example1():
    # 拿官方的例子改动一下
    def my_image_filter():
        conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv1_weights")
        conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
        conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]), name="conv2_weights")
        conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
        return None

    # First call creates one set of 4 variables.
    result1 = my_image_filter()
    # Another set of 4 variables is created in the second call.
    result2 = my_image_filter()
    # 获取所有的可训练变量
    vs = tf.trainable_variables()
    print 'There are %d train_able_variables in the Graph: ' % len(vs)
    for v in vs:
        print v


def example2():
    def conv_relu(kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
        return None

    def my_image_filter():
        # 按照下面的方式定义卷积层，非常直观，而且富有层次感
        with tf.variable_scope("conv1"):
            # Variables created here will be named "conv1/weights", "conv1/biases".
            relu1 = conv_relu([5, 5, 32, 32], [32])
        with tf.variable_scope("conv2"):
            # Variables created here will be named "conv2/weights", "conv2/biases".
            relu2 = conv_relu([5, 5, 32, 32], [32])
        return None

    with tf.variable_scope("image_filter") as scope:
        # 下面我们两次调用 my_image_filter 函数，但是由于引入了 变量共享机制
        # 可以看到我们只是创建了一遍网络结构。
        result1 = my_image_filter()
        scope.reuse_variables()
        result2 = my_image_filter()

    # 看看下面，完美地实现了变量共享！！！
    vs = tf.trainable_variables()
    print 'There are %d train_able_variables in the Graph: ' % len(vs)
    for v in vs:
        print v

    '''
    首先我们要确立一种 Graph 的思想。在 TensorFlow 中，我们定义一个变量，相当于往 Graph 中添加了一个节点。和普通的 python 函数不一样，
    在一般的函数中，我们对输入进行处理，然后返回一个结果，而函数里边定义的一些局部变量我们就不管了。
    但是在 TensorFlow 中，我们在函数里边创建了一个变量，就是往 Graph 中添加了一个节点。出了这个函数后，这个节点还是存在于 Graph 中的。
    在main函数中执行example1和example2，第二次会打印12条记录。单独执行example2，只有4条记录。正如上面所述，如果执行了example1，其实这些
    变量已经添加到graph中了，所以example2会输出12条记录（example1有8条记录）
    '''


# 通过集合获取变量
def test4_collection_vars():
    x = tf.constant(1.0, shape=[], name="x")  # 0D tensor
    k = tf.Variable(tf.constant(0.5, shape=[]), name="k")
    y = tf.multiply(x, k, name="y")

    t = tf.Variable(tf.constant(0.5, shape=[]), name="t", trainable=False)
    # v1, v2结果一致。只会输出k对应的变量信息. 后来增加了t变量，v1、v2就有了两个元素
    v1 = tf.global_variables()
    v2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print v1
    print v2

    v3 = tf.local_variables()  # 这个是空
    print v3

    v4 = tf.trainable_variables()  # 如果t没有设置trainable=False，v4也是返回k+T对应的变量
    print v4


if __name__ == "__main__":
    # test1()
    # test2()
    # example1()
    # print
    # example2()
    test4_collection_vars()
