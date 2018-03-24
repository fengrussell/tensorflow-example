# !/usr/bin/env python2
# -*- coding: utf-8 -*-


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


if __name__ == "__main__":
    # zip_test()
    # map_reduce_test()
    filter_test()
