# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# 主要验证slim计算loss和grad时，通过get_collection(key, scope)获取loss值是否时全部losses值。
# 结论：slim在添加loss值，使用tf.loss.add(loss)，这里其实调用tf.add_to_collection(key, loss)。获取loss时传入scope参数，这个时候
# 只获取scope下的key对应的losses。具体请看下面的模拟测试
def test_scope():
    with tf.name_scope('scope1') as scope1:
        v1 = tf.Variable([1], name='v1')
        # v3 = tf.Variable([1], name='v3')
        tf.add_to_collection('test', v1)
        # tf.add_to_collection('test', v3)

    with tf.name_scope('scope2') as scope2:
        v2 = tf.Variable([1], name='v2')
        tf.add_to_collection('test', v2)

    colls = tf.get_collection('test')
    # 没有scope参数，会返回test对应的所有的值
    # [<tf.Variable 'scope1/v1:0' shape=(1,) dtype=int32_ref>, <tf.Variable 'scope2/v2:0' shape=(1,) dtype=int32_ref>]
    print(colls)

    colls1 = tf.get_collection('test', scope1)
    # 传入scope1，只会返回v1
    # [<tf.Variable 'scope1/v1:0' shape=(1,) dtype=int32_ref>]
    print(colls1)

    colls2 = tf.get_collection('test', scope2)
    # [<tf.Variable 'scope2/v2:0' shape=(1,) dtype=int32_ref>]
    print(colls2)


if __name__ == "__main__":
    test_scope()