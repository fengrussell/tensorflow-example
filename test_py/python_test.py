# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import math

def zip_test():
    a = [1, 2, 3]
    b = [4, 5, 6]
    print zip(a, b)  # [(1, 4), (2, 5), (3, 6)]

    c = [(1, 4), (2, 5), (3, 6)]
    print zip(*c)  # [(1, 2, 3), (4, 5, 6)]

    c1 = [[1, 4], [2, 5], [3, 6]]
    print zip(*c1)  # [(1, 2, 3), (4, 5, 6)]

    d = [1, 4]
    e = [2, 5]
    f = [3, 6]
    print zip(d, e, f)  # *c/c1的拆解 [(1, 2, 3), (4, 5, 6)]


def map_reduce_test():
    def m(x):
        return x*x

    for x in map(m, [1, 2]):
        print x

    def r(a, b):
        return a * b
    print reduce(r, [1, 2, 3])


def filter_test():
    def f(x):
        return x % 2 == 0

    for x in filter(f, [1, 2, 3, 4, 5]):
        print x

    def no_empty(s):
        return s and s.strip()

    print list(filter(no_empty, ['A', '', 'B', None, 'C', '  ']))


def sorted_test():
    l = sorted([5, 4, 3, 1, 2])
    for i in l:
        print i

    l = sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower)
    for i in l:
        print i


def fun_test():
    def lazy_sum(*args):
        def sum():
            ax = 0
            for i in args:
                ax += i
            return ax
        return sum  # 如果返回的是sum(), 这个会立刻执行sum的运算返回结果。如果返回的是sum，只是返回的是一个function的引用，并不会执行sum函数

    s = lazy_sum(1, 2, 3)
    print s
    print s()


def closure_test():
    def count1():
        fs = []
        for i in range(1, 4):
            def f():
                return i * i

            fs.append(f)
        return fs

    f1, f2, f3 = count1()
    print f1(), f2(), f3()  # 结果全是9，而不是1、4、9。因为i在变化，最后计算时i=3

    def count2():
        fs = []
        for i in range(1, 4):
            def f(j):
                return j * j

            fs.append(f(i))
        return fs

    f1, f2, f3 = count2()
    print f1, f2, f3  # 1、4、9，给f传递一个参数。不过这种应该不是lazy方式加载了，执行fs.append(f(i))已经执行了。

    def count3():
        def f(j):
            def g():
                return j * j

            return g

        fs = []
        for i in range(1, 4):
            fs.append(f(i))  # f(i)立刻被执行，因此i的当前值被传入f()
        return fs

    f1, f2, f3 = count3()
    print f1(), f2(), f3()  # 1、4、9，这种方式是先把参数传给f，然后g延迟执行


if __name__ == "__main__":
    # zip_test()
    # map_reduce_test()
    # filter_test()
    # fun_test()
    closure_test()
